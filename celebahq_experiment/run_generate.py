"""
Generate CelebA-HQ images with trained LDM, compare VAE vs SPNN decoder.

Two modes:
  1. Generation: Sample latents from LDM (unconditional), decode with VAE and SPNN
  2. Reconstruction: Encode real test images with VAE, decode with both

Usage:
    python celebahq_experiment/run_generate.py --checkpoint <spnn_ckpt>
"""

import argparse
import math
import os
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from dataset import CelebAHQDataset
from cifar10_experiment.train_cifar10 import (
    SPNNAutoencoderConfigurable, _load_checkpoint)
from celebahq_experiment.train_spnn import CELEBAHQ_STAGES

SLDM_ROOT = os.path.join(os.path.dirname(__file__), '..',
    'simple-latent-diffusion-model-master', 'simple-latent-diffusion-model')
sys.path.insert(0, SLDM_ROOT)
from auto_encoder.models.variational_auto_encoder import VariationalAutoEncoder
from diffusion_model.models.latent_diffusion_model import LatentDiffusionModel
from diffusion_model.network.unet_wrapper import UnetWrapper
from diffusion_model.network.unet import Unet
from diffusion_model.sampler.ddim import DDIM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'celebahq_config.yaml')
VAE_PATH = os.path.join(os.path.dirname(__file__), 'models', 'celebahq_vae.pth')


def psnr(pred, target):
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return float("inf")
    return 10 * math.log10(4.0 / mse.item())


def load_ldm(args):
    """Load trained unconditional LatentDiffusionModel."""
    vae = VariationalAutoEncoder(CONFIG_PATH)
    vae = _load_checkpoint(vae, VAE_PATH)
    vae.eval().to(DEVICE)

    sampler = DDIM(CONFIG_PATH)
    network = UnetWrapper(Unet, CONFIG_PATH)  # unconditional
    ldm = LatentDiffusionModel(network, sampler, vae)
    ldm = _load_checkpoint(ldm, args.ldm_path)
    ldm.eval().to(DEVICE)

    return ldm, vae


def load_spnn(args):
    spnn = SPNNAutoencoderConfigurable(
        stages=CELEBAHQ_STAGES,
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
    """Sample latents from unconditional LDM, decode with VAE and SPNN."""
    out_dir = os.path.join(args.output_dir, "generation")
    os.makedirs(out_dir, exist_ok=True)

    print(f"\nGenerating {args.n_samples} unconditional samples...")

    # Generate latents (unconditional — no class label)
    latents = ldm(args.n_samples)  # [N, 3, 32, 32]

    vae_decoded = vae.decode(latents)
    spnn_decoded = spnn.decode(latents)

    p = psnr(spnn_decoded, vae_decoded)
    print(f"  PSNR(SPNN vs VAE): {p:.2f} dB")

    try:
        import lpips
        lpips_fn = lpips.LPIPS(net="vgg").to(DEVICE)
        lpips_fn.eval()
        lp = lpips_fn(spnn_decoded, vae_decoded).mean().item()
        print(f"  LPIPS(SPNN vs VAE): {lp:.4f}")
    except ImportError:
        pass

    # Save grid: row 1 = VAE decode, row 2 = SPNN decode
    vae_imgs = (vae_decoded.cpu() + 1) / 2
    spnn_imgs = (spnn_decoded.cpu() + 1) / 2
    grid = torch.cat([vae_imgs, spnn_imgs], dim=0)
    save_image(grid, os.path.join(out_dir, "generated.png"),
               nrow=args.n_samples, padding=2)
    print(f"  Saved to {out_dir}/generated.png (Row1: VAE, Row2: SPNN)")


@torch.no_grad()
def reconstruction_mode(vae, spnn, args):
    """Encode real CelebA-HQ test images with VAE, decode with both."""
    out_dir = os.path.join(args.output_dir, "reconstruction")
    os.makedirs(out_dir, exist_ok=True)

    dataset = CelebAHQDataset(img_size=128, split="test", n_test=1000)
    images = torch.stack([dataset[i] for i in range(args.n_recon)]).to(DEVICE)
    print(f"\nReconstruction mode: {images.shape[0]} test images")

    latents = vae.encode(images).mode()

    vae_decoded = vae.decode(latents)
    spnn_decoded = spnn.decode(latents)

    vae_psnr = psnr(vae_decoded, images)
    spnn_psnr = psnr(spnn_decoded, images)
    vae_vs_spnn = psnr(spnn_decoded, vae_decoded)

    print(f"  VAE recon PSNR:        {vae_psnr:.2f} dB")
    print(f"  SPNN recon PSNR:       {spnn_psnr:.2f} dB")
    print(f"  PSNR(SPNN vs VAE):     {vae_vs_spnn:.2f} dB")

    # SPNN roundtrip
    spnn_latent = spnn.encode(images)
    spnn_roundtrip = spnn.decode(spnn_latent)
    rt_psnr = psnr(spnn_roundtrip, images)
    print(f"  SPNN roundtrip PSNR:   {rt_psnr:.2f} dB")

    lat_mse = F.mse_loss(spnn_latent, latents).item()
    print(f"  Latent MSE(SPNN vs VAE): {lat_mse:.6f}")

    # Grid: Original | VAE | SPNN | SPNN roundtrip
    n = min(8, images.shape[0])
    grid = torch.cat([
        (images[:n].cpu() + 1) / 2,
        (vae_decoded[:n].cpu() + 1) / 2,
        (spnn_decoded[:n].cpu() + 1) / 2,
        (spnn_roundtrip[:n].cpu() + 1) / 2,
    ], dim=0)
    save_image(grid, os.path.join(out_dir, "reconstruction_grid.png"),
               nrow=n, padding=2)
    print(f"  Saved to {out_dir}/reconstruction_grid.png")
    print(f"  Rows: Original | VAE decode | SPNN decode | SPNN roundtrip")


def run(args):
    ldm, vae = load_ldm(args)
    spnn = load_spnn(args)
    generation_mode(ldm, vae, spnn, args)
    reconstruction_mode(vae, spnn, args)
    print(f"\nDone. Results in {args.output_dir}/")


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate CelebA-HQ with LDM, compare VAE vs SPNN")
    p.add_argument("--checkpoint", type=str, required=True,
                   help="SPNN checkpoint path")
    p.add_argument("--ldm_path", type=str,
                   default=os.path.join(os.path.dirname(__file__),
                                        'models', 'celebahq_ldm.pth'))
    p.add_argument("--n_samples", type=int, default=8)
    p.add_argument("--n_recon", type=int, default=32)
    p.add_argument("--output_dir", type=str,
                   default="celebahq_experiment/results")
    p.add_argument("--mix_type", type=str, default="cayley")
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--scale_bound", type=float, default=2.0)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)
