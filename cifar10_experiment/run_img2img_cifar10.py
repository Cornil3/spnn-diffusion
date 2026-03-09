"""
CIFAR-10 img2img: encode → noise → denoise → decode.

Compares VAE vs SPNN as the autoencoder in a diffusion img2img pipeline.
Tests whether SPNN can serve as a drop-in replacement for VAE encoding
and decoding in a standard diffusion workflow.
"""

import argparse
import math
import os
import sys

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm

# ── SPNN ──
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from cifar10_experiment.train_cifar10 import (
    SPNNAutoencoderConfigurable, CIFAR10_STAGES, _load_checkpoint)
from cifar10_experiment.train_ldm_with_spnn import SPNNAsVAE

# ── Simple LDM ──
SLDM_ROOT = os.path.join(os.path.dirname(__file__), '..',
    'simple-latent-diffusion-model-master', 'simple-latent-diffusion-model')
sys.path.insert(0, SLDM_ROOT)
from auto_encoder.models.variational_auto_encoder import VariationalAutoEncoder
from diffusion_model.models.latent_diffusion_model import LatentDiffusionModel
from diffusion_model.network.unet_wrapper import UnetWrapper
from diffusion_model.network.unet import Unet
from diffusion_model.sampler.ddim import DDIM
from helper.cond_encoder import ClassEncoder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONFIG_PATH = os.path.join(SLDM_ROOT, 'configs', 'cifar10_config.yaml')


# ═══════════════════════════════════════════════════════════
# Codec wrappers
# ═══════════════════════════════════════════════════════════

class SimpleVAECodec:
    def __init__(self, vae):
        self.vae = vae

    def encode(self, x):
        return self.vae.encode(x).mode()

    def decode(self, z):
        return self.vae.decode(z)


class SPNNCodec:
    def __init__(self, spnn):
        self.spnn = spnn

    def encode(self, x):
        return self.spnn.encode(x)

    def decode(self, z):
        return self.spnn.decode(z)


# ═══════════════════════════════════════════════════════════
# Model loading
# ═══════════════════════════════════════════════════════════

def load_spnn(args):
    """Load SPNN model from checkpoint."""
    print(f"Loading SPNN from {args.checkpoint}...")
    spnn = SPNNAutoencoderConfigurable(
        stages=CIFAR10_STAGES,
        mix_type=args.mix_type,
        hidden=args.hidden,
        scale_bound=args.scale_bound,
    )
    state = torch.load(args.checkpoint, map_location=DEVICE, weights_only=True)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    spnn.load_state_dict(state)
    spnn.eval().to(DEVICE)
    return spnn


def load_models(args):
    # Always load VAE (needed for VAE codec comparison)
    vae = VariationalAutoEncoder(CONFIG_PATH)
    vae = _load_checkpoint(vae, os.path.join(SLDM_ROOT, 'models', 'cifar_vae.pth'))
    vae.eval().to(DEVICE)
    for p in vae.parameters():
        p.requires_grad = False

    spnn = load_spnn(args)

    sampler = DDIM(CONFIG_PATH)
    cond_encoder = ClassEncoder(CONFIG_PATH)
    network = UnetWrapper(Unet, CONFIG_PATH, cond_encoder)

    # Build LDM with the matching auto_encoder for the checkpoint
    if args.spnn_ldm:
        # SPNN-trained LDM: UNet was trained on SPNN latents
        spnn_vae = SPNNAsVAE(spnn)
        ldm = LatentDiffusionModel(network, sampler, spnn_vae)
        ldm = _load_checkpoint(ldm, args.ldm_path)
        print(f"Loaded SPNN-trained LDM from {args.ldm_path}")
    else:
        # Original LDM: UNet was trained on VAE latents
        ldm = LatentDiffusionModel(network, sampler, vae)
        ldm = _load_checkpoint(ldm, args.ldm_path)
        print(f"Loaded original LDM from {args.ldm_path}")

    ldm.eval().to(DEVICE)
    for p in ldm.parameters():
        p.requires_grad = False

    return ldm, vae, spnn


# ═══════════════════════════════════════════════════════════
# DDIM denoise from noisy latent
# ═══════════════════════════════════════════════════════════

@torch.no_grad()
def ddim_denoise(ldm, z_t, class_label, start_tau, guidance_scale=3.0):
    """
    Denoise z_t from timestep start_tau back to clean latent z_0.
    """
    sampler = ldm.sampler
    network = ldm.network
    y_cond = torch.tensor([class_label], device=DEVICE)
    timesteps = sampler.timesteps.flip(0)

    # Start from the given tau index
    for i in range(start_tau, len(timesteps)):
        tau = len(timesteps) - 1 - i
        t = timesteps[i].to(DEVICE)

        # Classifier-free guidance
        eps_cond = network(x=z_t, t=t, y=y_cond)
        eps_uncond = network(x=z_t, t=t, y=y_cond, cond_drop_all=True)
        noise_pred = guidance_scale * eps_cond + (1 - guidance_scale) * eps_uncond

        # Estimate z_0
        alpha_t = sampler.ddim_alpha[tau].to(DEVICE)
        sqrt_one_minus_alpha = sampler.sqrt_one_minus_alpha_bar[tau].to(DEVICE)
        z_0_pred = (z_t - sqrt_one_minus_alpha * noise_pred) / alpha_t.sqrt()

        # DDIM step
        alpha_prev = sampler.alpha_bar_prev[tau].to(DEVICE)
        sigma = sampler.sigma[tau].to(DEVICE)
        dir_xt = (1.0 - alpha_prev - sigma ** 2).sqrt() * noise_pred
        noise = torch.randn_like(z_t) if sigma > 0 else 0
        z_t = alpha_prev.sqrt() * z_0_pred + dir_xt + sigma * noise

    return z_t


# ═══════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════

def psnr(pred, target):
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return float("inf")
    return 10 * math.log10(4.0 / mse.item())


def to_display(t):
    return ((t.cpu() + 1) / 2).clamp(0, 1)


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

@torch.no_grad()
def run(args):
    ldm, vae, spnn = load_models(args)

    codecs = {
        "VAE": SimpleVAECodec(vae),
        "SPNN": SPNNCodec(spnn),
    }

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    test_ds = datasets.CIFAR10(root='./data', train=False,
                                download=True, transform=transform)

    # Pick one image per class
    images_list, labels_list = [], []
    class_counts = [0] * 10
    for img, label in test_ds:
        if class_counts[label] < args.num_images_per_class:
            images_list.append(img)
            labels_list.append(label)
            class_counts[label] += 1
        if all(c >= args.num_images_per_class for c in class_counts):
            break

    images = torch.stack(images_list).to(DEVICE)
    print(f"Loaded {images.shape[0]} test images")

    sampler = ldm.sampler
    timesteps = sampler.timesteps.flip(0)
    num_steps = len(timesteps)

    # Find start step based on noise_strength
    start_tau = max(0, num_steps - int(num_steps * args.noise_strength))
    t_noise = timesteps[start_tau]
    alpha_bar_t = sampler.alpha_bar[t_noise].to(DEVICE)
    print(f"noise_strength={args.noise_strength}, start_tau={start_tau}, "
          f"t={t_noise.item()}, alpha_bar={alpha_bar_t.item():.4f}")

    out_dir = os.path.join("cifar10_experiment", "img2img_results",
                           f"noise{args.noise_strength}")
    os.makedirs(out_dir, exist_ok=True)

    all_metrics = {name: {"psnr": [], "mse": []} for name in codecs}

    for idx in range(images.shape[0]):
        x_gt = images[idx:idx + 1]
        class_label = labels_list[idx]

        for codec_name, codec in codecs.items():
            # 1. Encode
            z_0 = codec.encode(x_gt)

            # 2. Add noise
            eps = torch.randn_like(z_0)
            z_t = alpha_bar_t.sqrt() * z_0 + (1 - alpha_bar_t).sqrt() * eps

            # 3. Denoise
            z_clean = ddim_denoise(ldm, z_t, class_label, start_tau,
                                   guidance_scale=args.guidance_scale)

            # 4. Decode
            x_recon = codec.decode(z_clean).clamp(-1, 1)

            p = psnr(x_recon, x_gt)
            m = F.mse_loss(x_recon, x_gt).item()
            all_metrics[codec_name]["psnr"].append(p)
            all_metrics[codec_name]["mse"].append(m)
            print(f"  Image {idx+1}/{images.shape[0]} class={class_label} "
                  f"codec={codec_name}: PSNR={p:.2f} dB, MSE={m:.6f}")

        # Save grid: Original | VAE result | SPNN result
        grid_imgs = [to_display(x_gt[0])]
        for codec_name, codec in codecs.items():
            z_0 = codec.encode(x_gt)
            eps = torch.randn_like(z_0)
            # Use same seed for fair comparison
            torch.manual_seed(args.seed + idx)
            eps = torch.randn_like(z_0)
            z_t = alpha_bar_t.sqrt() * z_0 + (1 - alpha_bar_t).sqrt() * eps
            z_clean = ddim_denoise(ldm, z_t, class_label, start_tau,
                                   guidance_scale=args.guidance_scale)
            x_recon = codec.decode(z_clean).clamp(-1, 1)
            grid_imgs.append(to_display(x_recon[0]))

        grid = torch.stack(grid_imgs)
        save_image(grid, os.path.join(out_dir, f"img{idx:03d}_class{class_label}.png"),
                   nrow=3, padding=2)

    # Summary
    print(f"\n{'='*50}")
    print(f"img2img Summary (noise_strength={args.noise_strength})")
    print(f"{'='*50}")
    for codec_name in codecs:
        vals = all_metrics[codec_name]
        avg_psnr = sum(vals["psnr"]) / len(vals["psnr"])
        avg_mse = sum(vals["mse"]) / len(vals["mse"])
        print(f"  {codec_name:>5s}: avg PSNR = {avg_psnr:.2f} dB, avg MSE = {avg_mse:.6f}")

    print(f"\nResults saved to {out_dir}/")


def parse_args():
    p = argparse.ArgumentParser(
        description="CIFAR-10 img2img: encode → noise → denoise → decode")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--ldm_path", type=str,
                   default=os.path.join(SLDM_ROOT, 'models', 'cifar_ldm.pth'))
    p.add_argument("--spnn_ldm", action="store_true",
                   help="Set if ldm_path points to an SPNN-trained LDM checkpoint")
    p.add_argument("--noise_strength", type=float, default=0.5,
                   help="Fraction of diffusion steps to noise (0=no noise, 1=full noise)")
    p.add_argument("--num_images_per_class", type=int, default=1)
    p.add_argument("--guidance_scale", type=float, default=3.0)
    p.add_argument("--seed", type=int, default=42)
    # SPNN model args
    p.add_argument("--mix_type", type=str, default="cayley")
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--scale_bound", type=float, default=0.2)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(args.seed)
    run(args)
