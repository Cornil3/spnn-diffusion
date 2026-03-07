"""
DDNM inverse problems on CIFAR-10: VAE vs SPNN codec comparison.

At each reverse diffusion step the codec does decode → edit → encode.
SPNN's better cycle consistency should preserve more information across
the ~50 DDIM steps compared to the VAE.

Degradation operators (pixel space, scale-adjusted for 32x32):
  - Super-resolution 2x (32x32 -> 16x16 -> 32x32)
  - Random inpainting (50% mask)
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
# Degradation operators (pixel-space, 32x32 scale)
# ═══════════════════════════════════════════════════════════

class SuperResolution2x:
    """A(x) = avg_pool(x, 2), A_pinv(y) = nearest_upsample(y, 2)."""
    name = "sr2x"

    def __init__(self, img_size):
        self.img_size = img_size

    def A(self, x):
        return F.avg_pool2d(x, 2)

    def A_pinv(self, y):
        return F.interpolate(y, scale_factor=2, mode="nearest")

    def null_space_project(self, x_0, y):
        return self.A_pinv(y) + x_0 - self.A_pinv(self.A(x_0))


class RandomInpainting:
    """50% random pixel mask."""
    name = "inpaint"

    def __init__(self, img_size, mask_ratio=0.5, seed=42):
        gen = torch.Generator().manual_seed(seed)
        self.mask = (torch.rand(1, 1, img_size, img_size,
                                generator=gen) > mask_ratio).float()

    def _mask(self, x):
        return self.mask.to(x.device)

    def A(self, x):
        return x * self._mask(x)

    def A_pinv(self, y):
        return y

    def null_space_project(self, x_0, y):
        m = self._mask(x_0)
        return y * m + x_0 * (1 - m)


DEGRADATION_REGISTRY = {
    "sr2x": SuperResolution2x,
    "inpaint": RandomInpainting,
}


# ═══════════════════════════════════════════════════════════
# Codec wrappers
# ═══════════════════════════════════════════════════════════

class SimpleVAECodec:
    """Codec for Simple LDM VAE."""
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


class HybridCodec:
    """SPNN decode + VAE encode — keeps latents in UNet's expected distribution."""
    def __init__(self, spnn, vae):
        self.spnn = spnn
        self.vae = vae

    def encode(self, x):
        return self.vae.encode(x).mode()

    def decode(self, z):
        return self.spnn.decode(z)


# ═══════════════════════════════════════════════════════════
# Model loading
# ═══════════════════════════════════════════════════════════

def load_models(args):
    """Load LDM (UNet + DDIM + VAE) and SPNN."""
    # VAE
    vae = VariationalAutoEncoder(CONFIG_PATH)
    vae = _load_checkpoint(vae, os.path.join(SLDM_ROOT, 'models', 'cifar_vae.pth'))
    vae.eval().to(DEVICE)
    for p in vae.parameters():
        p.requires_grad = False

    # LDM
    sampler = DDIM(CONFIG_PATH)
    cond_encoder = ClassEncoder(CONFIG_PATH)
    network = UnetWrapper(Unet, CONFIG_PATH, cond_encoder)
    ldm = LatentDiffusionModel(network, sampler, vae)
    ldm = _load_checkpoint(ldm, args.ldm_path)
    ldm.eval().to(DEVICE)
    for p in ldm.parameters():
        p.requires_grad = False

    # SPNN
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

    return ldm, vae, spnn


# ═══════════════════════════════════════════════════════════
# DDNM reverse process (latent-space with codec round-trip)
# ═══════════════════════════════════════════════════════════

@torch.no_grad()
def ddnm_sample(ldm, codec, degradation, y, class_label,
                guidance_scale=3.0):
    """
    Latent DDNM using the trained LDM's DDIM sampler.

    At each reverse step:
      1. Predict noise → estimate clean latent z_0
      2. Decode z_0 to pixel space via codec
      3. Null-space projection (data consistency in pixel space)
      4. Re-encode corrected pixels back to latent space via codec
      5. DDIM step to z_{t-1}

    Steps 2-4 are the codec round-trip where SPNN's cycle consistency helps.
    """
    B = y.shape[0]
    sampler = ldm.sampler
    network = ldm.network
    latent_shape = (B, *ldm.image_shape)

    # Class conditioning
    y_cond = torch.tensor([class_label], device=DEVICE)

    # Start from noise in latent space
    z_t = torch.randn(latent_shape, device=DEVICE)

    # DDIM timesteps (high to low)
    timesteps = sampler.timesteps.flip(0)

    for i, t in enumerate(tqdm(timesteps, desc="DDNM steps", leave=False)):
        tau = len(timesteps) - 1 - i
        t_tensor = t.to(DEVICE)

        # 1. Predict noise with classifier-free guidance
        eps_cond = network(x=z_t, t=t_tensor, y=y_cond)
        eps_uncond = network(x=z_t, t=t_tensor, y=y_cond, cond_drop_all=True)
        noise_pred = guidance_scale * eps_cond + (1 - guidance_scale) * eps_uncond

        # 2. Estimate clean latent z_0
        alpha_t = sampler.ddim_alpha[tau].to(DEVICE)
        sqrt_one_minus_alpha = sampler.sqrt_one_minus_alpha_bar[tau].to(DEVICE)
        z_0_pred = (z_t - sqrt_one_minus_alpha * noise_pred) / alpha_t.sqrt()

        # 3. Decode to pixel space via codec
        x_0_pred = codec.decode(z_0_pred)

        # 4. Null-space projection (data consistency in pixel space)
        x_0_corrected = degradation.null_space_project(x_0_pred, y)

        # 5. Re-encode to latent space via codec
        z_0_corrected = codec.encode(x_0_corrected)

        # 6. DDIM step to z_{t-1}
        alpha_prev = sampler.alpha_bar_prev[tau].to(DEVICE)
        sigma = sampler.sigma[tau].to(DEVICE)
        dir_xt = (1.0 - alpha_prev - sigma ** 2).sqrt() * noise_pred
        noise = torch.randn_like(z_t) if sigma > 0 else 0
        z_t = alpha_prev.sqrt() * z_0_corrected + dir_xt + sigma * noise

    # Final decode to pixels
    return codec.decode(z_t)


# ═══════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════

def psnr(pred, target):
    """PSNR in dB, images in [-1, 1]."""
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return float("inf")
    return 10 * math.log10(4.0 / mse.item())


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

def run(args):
    ldm, vae, spnn = load_models(args)

    codecs = {
        "VAE": SimpleVAECodec(vae),
        "SPNN": SPNNCodec(spnn),
        "Hybrid": HybridCodec(spnn, vae),
    }

    # Load test images
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    test_ds = datasets.CIFAR10(root='./data', train=False,
                               download=True, transform=transform)

    # Select images (one per class if possible)
    images_list = []
    labels_list = []
    class_counts = [0] * 10
    for img, label in test_ds:
        if class_counts[label] < args.num_images_per_class:
            images_list.append(img)
            labels_list.append(label)
            class_counts[label] += 1
        if all(c >= args.num_images_per_class for c in class_counts):
            break

    images = torch.stack(images_list).to(DEVICE)
    labels = labels_list
    print(f"Loaded {images.shape[0]} test images ({images.shape[1:]})")

    for problem_name in args.problems:
        print(f"\n{'='*60}")
        print(f"Problem: {problem_name}")
        print(f"{'='*60}")

        deg_cls = DEGRADATION_REGISTRY[problem_name]
        degradation = deg_cls(32)

        out_dir = os.path.join("cifar10_experiment", "ddnm_results", problem_name)
        os.makedirs(out_dir, exist_ok=True)

        all_metrics = {name: {"psnr": [], "mse": []} for name in codecs}

        for img_idx in range(images.shape[0]):
            x_gt = images[img_idx:img_idx + 1]
            class_label = labels[img_idx]

            # Create degraded measurement
            y = degradation.A(x_gt)

            results = {}
            for codec_name, codec in codecs.items():
                print(f"  Image {img_idx + 1}/{images.shape[0]}, "
                      f"class={class_label}, codec={codec_name}...")
                x_recon = ddnm_sample(
                    ldm, codec, degradation, y,
                    class_label=class_label,
                    guidance_scale=args.guidance_scale,
                )
                x_recon = x_recon.clamp(-1, 1)
                results[codec_name] = x_recon

                p = psnr(x_recon, x_gt)
                m = F.mse_loss(x_recon, x_gt).item()
                all_metrics[codec_name]["psnr"].append(p)
                all_metrics[codec_name]["mse"].append(m)
                print(f"    PSNR: {p:.2f} dB, MSE: {m:.6f}")

            # Save grid: Degraded | VAE result | SPNN result | Ground truth
            if problem_name == "sr2x":
                y_display = F.interpolate(y, scale_factor=2, mode="nearest")
            else:
                y_display = y

            grid_images = [
                (y_display[0].cpu() + 1) / 2,
                (results["VAE"][0].cpu() + 1) / 2,
                (results["SPNN"][0].cpu() + 1) / 2,
                (results["Hybrid"][0].cpu() + 1) / 2,
                (x_gt[0].cpu() + 1) / 2,
            ]
            grid = torch.stack(grid_images)
            save_image(grid, os.path.join(out_dir, f"img{img_idx:03d}.png"),
                       nrow=5, padding=2)

        # Summary
        print(f"\n--- {problem_name} Summary ---")
        for codec_name in codecs:
            vals = all_metrics[codec_name]
            avg_psnr = sum(vals["psnr"]) / len(vals["psnr"])
            avg_mse = sum(vals["mse"]) / len(vals["mse"])
            print(f"  {codec_name:>5s}: avg PSNR = {avg_psnr:.2f} dB, "
                  f"avg MSE = {avg_mse:.6f}")

    print("\nDone. Results saved to cifar10_experiment/ddnm_results/")


def parse_args():
    p = argparse.ArgumentParser(
        description="DDNM on CIFAR-10: VAE vs SPNN codec comparison")
    p.add_argument("--checkpoint", type=str, required=True,
                   help="SPNN model checkpoint path")
    p.add_argument("--ldm_path", type=str,
                   default=os.path.join(SLDM_ROOT, 'models', 'cifar_ldm.pth'),
                   help="Trained LDM checkpoint path")
    p.add_argument("--problems", nargs="+", default=["sr2x", "inpaint"],
                   choices=list(DEGRADATION_REGISTRY.keys()))
    p.add_argument("--num_images_per_class", type=int, default=1,
                   help="Number of test images per class")
    p.add_argument("--guidance_scale", type=float, default=3.0)
    # SPNN model args
    p.add_argument("--mix_type", type=str, default="cayley")
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--scale_bound", type=float, default=2.0)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)
