"""
Generate CIFAR-10 images with trained LDM, compare VAE vs SPNN decoder.

The LDM generates latents (3x16x16) via diffusion, then we decode with
both the VAE decoder and SPNN decoder to compare quality.

Two modes:
  1. Generation: Sample latents from LDM, decode with VAE and SPNN
  2. Reconstruction: Encode real test images with VAE, decode with both

For cycle-consistency and Penrose diagnostics, use run_test_cycles_cifar10.py.
"""

import argparse
import math
import os
import sys

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

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
CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']


def psnr(pred, target):
    """PSNR in dB, images in [-1, 1]."""
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return float("inf")
    return 10 * math.log10(4.0 / mse.item())


def load_ldm(args):
    """Load trained LatentDiffusionModel (VAE + UNet)."""
    vae = VariationalAutoEncoder(CONFIG_PATH)
    vae = _load_checkpoint(vae, os.path.join(SLDM_ROOT, 'models', 'cifar_vae.pth'))
    vae.eval().to(DEVICE)

    sampler = DDIM(CONFIG_PATH)
    cond_encoder = ClassEncoder(CONFIG_PATH)
    network = UnetWrapper(Unet, CONFIG_PATH, cond_encoder)
    ldm = LatentDiffusionModel(network, sampler, vae)
    ldm = _load_checkpoint(ldm, args.ldm_path)
    ldm.eval().to(DEVICE)

    return ldm, vae


def load_spnn(args):
    """Load trained SPNN checkpoint."""
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


@torch.no_grad()
def generation_mode(ldm, vae, spnn, args):
    """Sample latents from LDM, decode with VAE and SPNN."""
    out_dir = os.path.join(args.output_dir, "generation")
    os.makedirs(out_dir, exist_ok=True)

    all_lpips_vals = []
    all_psnr_vals = []

    try:
        import lpips
        lpips_fn = lpips.LPIPS(net="vgg").to(DEVICE)
        lpips_fn.eval()
    except ImportError:
        lpips_fn = None

    for class_idx in range(10):
        class_name = CIFAR10_CLASSES[class_idx]
        print(f"\nGenerating class {class_idx}: {class_name}")

        # Generate latents from LDM (forward returns latents, not pixels)
        y = torch.tensor([class_idx], device=DEVICE)
        latents = ldm(args.n_samples, gamma=args.gamma, y=y)  # [N, 3, 16, 16]

        # Decode same latents with both decoders
        vae_decoded = vae.decode(latents)
        spnn_decoded = spnn.decode(latents)

        # Metrics between the two decodes
        p = psnr(spnn_decoded, vae_decoded)
        all_psnr_vals.append(p)
        print(f"  PSNR(SPNN vs VAE): {p:.2f} dB")

        if lpips_fn is not None:
            up_spnn = F.interpolate(spnn_decoded, size=64, mode='bilinear',
                                    align_corners=False)
            up_vae = F.interpolate(vae_decoded, size=64, mode='bilinear',
                                   align_corners=False)
            lp = lpips_fn(up_spnn, up_vae).mean().item()
            all_lpips_vals.append(lp)
            print(f"  LPIPS(SPNN vs VAE): {lp:.4f}")

        # Save grid: VAE decode | SPNN decode
        vae_imgs = (vae_decoded.cpu() + 1) / 2
        spnn_imgs = (spnn_decoded.cpu() + 1) / 2
        grid = torch.cat([vae_imgs, spnn_imgs], dim=0)
        save_image(grid, os.path.join(out_dir, f"class{class_idx}_{class_name}.png"),
                   nrow=args.n_samples, padding=2)

    # Summary
    print(f"\n{'='*50}")
    print("Generation Summary")
    print(f"{'='*50}")
    avg_psnr = sum(all_psnr_vals) / len(all_psnr_vals)
    print(f"  Avg PSNR(SPNN vs VAE): {avg_psnr:.2f} dB")
    if all_lpips_vals:
        avg_lpips = sum(all_lpips_vals) / len(all_lpips_vals)
        print(f"  Avg LPIPS(SPNN vs VAE): {avg_lpips:.4f}")
    print(f"  Grid rows: VAE decode | SPNN decode")


@torch.no_grad()
def reconstruction_mode(vae, spnn, args):
    """Encode real CIFAR-10 test images with VAE, decode with both."""
    out_dir = os.path.join(args.output_dir, "reconstruction")
    os.makedirs(out_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    test_ds = datasets.CIFAR10(root='./data', train=False,
                               download=True, transform=transform)

    images = torch.stack([test_ds[i][0] for i in range(args.n_recon)]).to(DEVICE)
    print(f"\nReconstruction mode: {images.shape[0]} test images")

    # Encode with VAE
    latents = vae.encode(images).mode()  # 3x16x16

    # Decode with both
    vae_decoded = vae.decode(latents)
    spnn_decoded = spnn.decode(latents)

    # Metrics vs original
    vae_psnr = psnr(vae_decoded, images)
    spnn_psnr = psnr(spnn_decoded, images)
    vae_vs_spnn_psnr = psnr(spnn_decoded, vae_decoded)

    print(f"  VAE recon PSNR:        {vae_psnr:.2f} dB")
    print(f"  SPNN recon PSNR:       {spnn_psnr:.2f} dB")
    print(f"  PSNR(SPNN vs VAE):     {vae_vs_spnn_psnr:.2f} dB")

    # Roundtrip: SPNN encode + decode
    spnn_latent = spnn.encode(images)
    spnn_roundtrip = spnn.decode(spnn_latent)
    rt_psnr = psnr(spnn_roundtrip, images)
    print(f"  SPNN roundtrip PSNR:   {rt_psnr:.2f} dB")

    # Latent alignment
    lat_mse = F.mse_loss(spnn_latent, latents).item()
    print(f"  Latent MSE(SPNN vs VAE): {lat_mse:.6f}")

    # Save grid: Original | VAE decode | SPNN decode | SPNN roundtrip
    n = min(8, images.shape[0])
    grid = torch.cat([
        (images[:n].cpu() + 1) / 2,
        (vae_decoded[:n].cpu() + 1) / 2,
        (spnn_decoded[:n].cpu() + 1) / 2,
        (spnn_roundtrip[:n].cpu() + 1) / 2,
    ], dim=0)
    save_image(grid, os.path.join(out_dir, "reconstruction_grid.png"),
               nrow=n, padding=2)
    print(f"  Saved grid to {out_dir}/reconstruction_grid.png")
    print(f"  Rows: Original | VAE decode | SPNN decode | SPNN roundtrip")


def run(args):
    ldm, vae = load_ldm(args)
    spnn = load_spnn(args)

    generation_mode(ldm, vae, spnn, args)
    reconstruction_mode(vae, spnn, args)

    print(f"\nDone. Results in {args.output_dir}/")


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate CIFAR-10 with LDM, compare VAE vs SPNN decode")
    p.add_argument("--checkpoint", type=str, required=True,
                   help="SPNN model checkpoint path")
    p.add_argument("--ldm_path", type=str,
                   default=os.path.join(SLDM_ROOT, 'models', 'cifar_ldm.pth'),
                   help="Trained LDM checkpoint path")
    p.add_argument("--n_samples", type=int, default=8,
                   help="Samples per class for generation")
    p.add_argument("--n_recon", type=int, default=32,
                   help="Number of test images for reconstruction")
    p.add_argument("--gamma", type=float, default=3.0,
                   help="Classifier-free guidance scale")
    p.add_argument("--output_dir", type=str,
                   default="cifar10_experiment/results")
    # SPNN model args (must match checkpoint)
    p.add_argument("--mix_type", type=str, default="cayley")
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--scale_bound", type=float, default=2.0)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)
