"""
Cycle-consistency test for CIFAR-10 SPNN: per-image Penrose checks +
multi-cycle encode→decode comparison (Simple LDM VAE vs SPNN).

Mirrors run_test_cycles.py from the CelebHQ experiment.
"""

import argparse
import math
import os
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm
import wandb

# ── SPNN ──
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from cifar10_experiment.train_cifar10 import (
    SPNNAutoencoderConfigurable, CIFAR10_STAGES, _load_checkpoint)
from diagnostics import penrose_check, print_penrose_metrics

# ── Simple LDM ──
SLDM_ROOT = os.path.join(os.path.dirname(__file__), '..',
    'simple-latent-diffusion-model-master', 'simple-latent-diffusion-model')
sys.path.insert(0, SLDM_ROOT)
from auto_encoder.models.variational_auto_encoder import VariationalAutoEncoder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONFIG_PATH = os.path.join(SLDM_ROOT, 'configs', 'cifar10_config.yaml')


def to_display(tensor):
    return ((tensor.cpu() + 1) / 2).clamp(0, 1)


def calc_psnr(a, b):
    m = F.mse_loss(a, b).item()
    return 10 * math.log10(4.0 / m) if m > 0 else float("inf")


def calc_mse(a, b):
    return F.mse_loss(a, b).item()


# ──────────────────────── Load models ────────────────────────

def load_simple_vae():
    print("Loading Simple LDM VAE...")
    vae = VariationalAutoEncoder(CONFIG_PATH)
    model_path = os.path.join(SLDM_ROOT, 'models', 'cifar_vae.pth')
    vae = _load_checkpoint(vae, model_path)
    vae.eval().to(DEVICE)
    for p in vae.parameters():
        p.requires_grad = False
    return vae


def load_spnn(ckpt_path, mix_type, hidden, scale_bound):
    print(f"Loading SPNN from {ckpt_path}...")
    spnn = SPNNAutoencoderConfigurable(
        stages=CIFAR10_STAGES,
        mix_type=mix_type,
        hidden=hidden,
        scale_bound=scale_bound,
    )
    state = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    spnn.load_state_dict(state)
    spnn.eval().to(DEVICE)
    return spnn


# ──────────────────────── Single cycle functions ────────────────────────

@torch.no_grad()
def vae_cycle(vae, x):
    """Simple LDM VAE encode -> decode. Lossy each pass."""
    z = vae.encode(x).mode()
    return vae.decode(z)


@torch.no_grad()
def spnn_cycle(spnn, x):
    """
    SPNN encode -> decode via pseudo-inverse (r network).
    No side-channel latents — decode() uses the r network to estimate x1.
    This is the realistic path for DDNM / diffusion pipelines.
    """
    latent = spnn.encode(x)
    return spnn.decode(latent)


# ──────────────────────── Main ────────────────────────

def run_test(args):
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device:     {DEVICE}")
    print(f"Cycles:     {args.num_cycles}\n")

    vae = load_simple_vae()
    spnn = load_spnn(args.checkpoint, args.mix_type, args.hidden, args.scale_bound)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    dataset = datasets.CIFAR10(root='./data', train=False,
                               download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    print(f"Test set: {len(dataset)} images\n")

    test_sample_dir = os.path.join(args.sample_dir, "test")
    os.makedirs(test_sample_dir, exist_ok=True)

    # ── Aggregate metrics per cycle ──
    cycle_metrics = {c: [] for c in range(1, args.num_cycles + 1)}
    all_penrose = []

    for img_idx, (original, _label) in enumerate(tqdm(loader, desc="Testing")):
        if args.num_test_images > 0 and img_idx >= args.num_test_images:
            break
        original = original.to(DEVICE)

        # ── Penrose checks ──
        with torch.no_grad():
            spnn_latent = spnn.encode(original)
        p_metrics = penrose_check(spnn, original, spnn_latent, DEVICE)
        all_penrose.append(p_metrics)

        # ── Cycle test: VAE vs SPNN (pseudo-inverse) ──
        vae_x = original.clone()
        spnn_x = original.clone()

        vae_imgs = [original.clone()]
        spnn_imgs = [original.clone()]

        for c in range(1, args.num_cycles + 1):
            vae_x = vae_cycle(vae, vae_x)
            spnn_x = spnn_cycle(spnn, spnn_x)

            vae_imgs.append(vae_x.clone())
            spnn_imgs.append(spnn_x.clone())

            cycle_metrics[c].append((
                calc_mse(vae_x, original),
                calc_psnr(vae_x, original),
                calc_mse(spnn_x, original),
                calc_psnr(spnn_x, original),
            ))

        # ── Save visual grid for first few images ──
        if img_idx < args.num_save_images:
            row_vae = torch.cat([to_display(img) for img in vae_imgs], dim=0)
            row_spnn = torch.cat([to_display(img) for img in spnn_imgs], dim=0)
            grid = torch.cat([row_vae, row_spnn], dim=0)
            grid_path = os.path.join(test_sample_dir, f"test_cycles_{img_idx:03d}.png")
            save_image(grid, grid_path, nrow=args.num_cycles + 1, padding=4, pad_value=1.0)
            wandb.log({"test/cycle_grids": wandb.Image(grid_path,
                       caption=f"Image {img_idx} — Row1: VAE, Row2: SPNN (pinv)")})

            # ── Summary: original | VAE@last | SPNN@last ──
            summary = torch.cat([
                to_display(original),
                to_display(vae_imgs[-1]),
                to_display(spnn_imgs[-1]),
            ], dim=0)
            summary_path = os.path.join(test_sample_dir, f"test_summary_{img_idx:03d}.png")
            save_image(summary, summary_path, nrow=3, padding=4, pad_value=1.0)
            wandb.log({"test/summaries": wandb.Image(summary_path,
                       caption=f"Image {img_idx} — Orig | VAE@{args.num_cycles} | SPNN@{args.num_cycles}")})

    # ── Average Penrose metrics ──
    avg_penrose = {}
    for key in all_penrose[0]:
        avg_penrose[key] = sum(p[key] for p in all_penrose) / len(all_penrose)
    print("Penrose pseudo-inverse diagnostics (averaged over test set):")
    print_penrose_metrics(avg_penrose)
    wandb.log({"test/" + k: v for k, v in avg_penrose.items()})
    print()

    # ── Average cycle metrics ──
    header = (f"{'Cycle':<7} "
              f"{'VAE MSE':<13} {'VAE PSNR':<13} "
              f"{'SPNN MSE':<14} {'SPNN PSNR':<13}")
    print(header)
    print("-" * len(header))

    for c in range(1, args.num_cycles + 1):
        vals = cycle_metrics[c]
        n = len(vals)
        avg_vae_mse = sum(v[0] for v in vals) / n
        avg_vae_psnr = sum(v[1] for v in vals) / n
        avg_spnn_mse = sum(v[2] for v in vals) / n
        avg_spnn_psnr = sum(v[3] for v in vals) / n

        wandb.log({
            "test/cycle": c,
            "test/vae_mse": avg_vae_mse,
            "test/vae_psnr": avg_vae_psnr,
            "test/spnn_mse": avg_spnn_mse,
            "test/spnn_psnr": avg_spnn_psnr,
        })

        print(f"{c:<7} "
              f"{avg_vae_mse:<13.6f} {avg_vae_psnr:<13.2f} "
              f"{avg_spnn_mse:<14.2e} {avg_spnn_psnr:<13.2f}")

    print(f"\nAveraged over {len(dataset)} test images.")
    print(f"Saved visual grids to {test_sample_dir}/test_cycles_*.png")


# ──────────────────────── Latent Alignment Check ────────────────────────

@torch.no_grad()
def latent_alignment_check_cifar10(spnn, vae, dataloader):
    """
    Compare SPNN.encode(x) vs VAE.encode(x) over the test set.
    Adapted for Simple LDM VAE (no scaling_factor, .mode() API).
    """
    spnn.eval()

    total_mse = 0.0
    total_cos = 0.0
    n = 0
    num_el = None

    for images, _labels in tqdm(dataloader, desc="Latent alignment"):
        images = images.to(DEVICE)

        z_spnn = spnn.encode(images)
        z_vae = vae.encode(images).mode()  # Simple LDM API: no scaling_factor

        bs = images.size(0)
        if num_el is None:
            num_el = z_spnn[0].numel()

        total_mse += F.mse_loss(z_spnn, z_vae, reduction="sum").item()
        cos = F.cosine_similarity(
            z_spnn.view(bs, -1), z_vae.view(bs, -1), dim=1
        ).sum().item()
        total_cos += cos
        n += bs

    metrics = {
        "latent_align/mse": total_mse / (n * num_el),
        "latent_align/cosine_sim": total_cos / n,
        "latent_align/num_samples": n,
    }

    print(f"\n{'='*50}")
    print(f"Latent Alignment Check ({n} samples)")
    print(f"{'='*50}")
    print(f"  MSE(z_spnn, z_vae):    {metrics['latent_align/mse']:.6f}")
    print(f"  Cosine similarity:     {metrics['latent_align/cosine_sim']:.6f}")
    print(f"  (cosine=1.0 means perfect alignment)")

    return metrics


@torch.no_grad()
def cross_decode_check_cifar10(spnn, vae, dataloader, output_dir, num_images=5):
    """
    Encode with SPNN, decode with VAE (and vice versa).
    Adapted for Simple LDM VAE (no scaling_factor, .mode()/.decode() API).

    Saves grid: Original | VAE(VAE(x)) | VAE(SPNN(x)) | SPNN(SPNN(x))
    """
    spnn.eval()
    os.makedirs(output_dir, exist_ok=True)

    count = 0
    all_mse = []

    print(f"\n{'='*50}")
    print(f"Cross-Decode Check ({num_images} images)")
    print(f"{'='*50}")
    print(f"  Grid: Original | VAE(VAE(x)) | VAE(SPNN(x)) | SPNN(SPNN(x))")
    print()

    for images, _labels in dataloader:
        images = images.to(DEVICE)
        for i in range(images.size(0)):
            if count >= num_images:
                break

            x = images[i:i+1]

            # VAE encode + decode (baseline)
            z_vae = vae.encode(x).mode()
            vae_recon = vae.decode(z_vae)

            # SPNN encode
            z_spnn = spnn.encode(x)

            # Cross-decode: VAE decodes SPNN's latent (no scaling needed)
            cross_recon = vae.decode(z_spnn)

            # SPNN roundtrip
            spnn_recon = spnn.decode(z_spnn)

            # Per-image metrics
            vae_mse = F.mse_loss(vae_recon, x).item()
            cross_mse = F.mse_loss(cross_recon, x).item()
            spnn_mse = F.mse_loss(spnn_recon, x).item()
            z_mse = F.mse_loss(z_spnn, z_vae).item()
            all_mse.append((vae_mse, cross_mse))

            print(f"  Image {count}: VAE recon MSE={vae_mse:.6f}  "
                  f"Cross-decode MSE={cross_mse:.6f}  "
                  f"SPNN roundtrip MSE={spnn_mse:.6f}  "
                  f"Latent MSE={z_mse:.6f}")

            # Save grid
            grid = torch.cat([
                to_display(x), to_display(vae_recon),
                to_display(cross_recon), to_display(spnn_recon),
            ], dim=0)
            path = os.path.join(output_dir, f"cross_decode_{count:03d}.png")
            save_image(grid, path, nrow=4, padding=4, pad_value=1.0)
            count += 1

        if count >= num_images:
            break

    print(f"\n  Saved {count} grids to {output_dir}/")
    print(f"  If cross-decode (col 3) looks good, SPNN latents are diffusion-compatible.")
    return all_mse


def run_latent_diagnostics(args):
    """Run latent alignment and cross-decode checks for CIFAR-10."""
    vae = load_simple_vae()
    spnn = load_spnn(args.checkpoint, args.mix_type, args.hidden, args.scale_bound)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    dataset = datasets.CIFAR10(root='./data', train=False,
                               download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    latent_alignment_check_cifar10(spnn, vae, loader)

    diag_dir = os.path.join(args.sample_dir, "latent_diagnostics")
    cross_decode_check_cifar10(spnn, vae, loader, diag_dir, num_images=5)


def parse_args():
    p = argparse.ArgumentParser(
        description="CIFAR-10 SPNN cycle-consistency test (mirrors run_test_cycles.py)")
    p.add_argument("--checkpoint", type=str, required=True,
                   help="SPNN model checkpoint path")
    p.add_argument("--num_cycles", type=int, default=5)
    p.add_argument("--num_test_images", type=int, default=0,
                   help="Limit number of test images (0=all)")
    p.add_argument("--num_save_images", type=int, default=30)
    p.add_argument("--sample_dir", type=str,
                   default="cifar10_experiment/samples")
    # SPNN model args (must match checkpoint)
    p.add_argument("--mix_type", type=str, default="cayley")
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--scale_bound", type=float, default=2.0)
    p.add_argument("--diagnostics_only", action="store_true",
                   help="Run only latent alignment + cross-decode (no wandb)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.diagnostics_only:
        run_latent_diagnostics(args)
    else:
        wandb.init(project="spnn-cifar10", config=vars(args))
        run_test(args)
        run_latent_diagnostics(args)
        wandb.finish()
