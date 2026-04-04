"""
Cycle-consistency test for CelebA-HQ SPNN: per-image Penrose checks +
multi-cycle encode->decode comparison (Simple LDM VAE vs SPNN).

Usage:
    python celebahq_experiment/run_test_cycles.py --checkpoint <path>
"""

import argparse
import math
import os
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
import wandb

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from dataset import CelebAHQDataset
from diagnostics import penrose_check, print_penrose_metrics
from cifar10_experiment.train_cifar10 import (
    SPNNAutoencoderConfigurable, _load_checkpoint)
from celebahq_experiment.train_spnn import CELEBAHQ_STAGES

SLDM_ROOT = os.path.join(os.path.dirname(__file__), '..',
    'simple-latent-diffusion-model-master', 'simple-latent-diffusion-model')
sys.path.insert(0, SLDM_ROOT)
from auto_encoder.models.variational_auto_encoder import VariationalAutoEncoder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'celebahq_config.yaml')
VAE_PATH = os.path.join(os.path.dirname(__file__), 'models', 'celebahq_vae.pth')


def to_display(tensor):
    return ((tensor.cpu() + 1) / 2).clamp(0, 1)


def calc_psnr(a, b):
    m = F.mse_loss(a, b).item()
    return 10 * math.log10(4.0 / m) if m > 0 else float("inf")


def calc_mse(a, b):
    return F.mse_loss(a, b).item()


def load_vae():
    vae = VariationalAutoEncoder(CONFIG_PATH)
    vae = _load_checkpoint(vae, VAE_PATH)
    vae.eval().to(DEVICE)
    for p in vae.parameters():
        p.requires_grad = False
    return vae


def load_spnn(ckpt_path, mix_type, hidden, scale_bound):
    spnn = SPNNAutoencoderConfigurable(
        stages=CELEBAHQ_STAGES,
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


@torch.no_grad()
def vae_cycle(vae, x):
    z = vae.encode(x).mode()
    return vae.decode(z)


@torch.no_grad()
def spnn_cycle(spnn, x):
    return spnn.decode(spnn.encode(x))


def run_test(args):
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device:     {DEVICE}")
    print(f"Cycles:     {args.num_cycles}\n")

    vae = load_vae()
    spnn = load_spnn(args.checkpoint, args.mix_type, args.hidden, args.scale_bound)

    dataset = CelebAHQDataset(img_size=128, split="test", n_test=1000)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    print(f"Test set: {len(dataset)} images\n")

    test_sample_dir = os.path.join(args.sample_dir, "test")
    os.makedirs(test_sample_dir, exist_ok=True)

    cycle_metrics = {c: [] for c in range(1, args.num_cycles + 1)}
    all_penrose = []

    for img_idx, original in enumerate(tqdm(loader, desc="Testing")):
        if args.num_test_images > 0 and img_idx >= args.num_test_images:
            break
        original = original.to(DEVICE)

        with torch.no_grad():
            spnn_latent = spnn.encode(original)
        p_metrics = penrose_check(spnn, original, spnn_latent, DEVICE)
        all_penrose.append(p_metrics)

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

        if img_idx < args.num_save_images:
            row_vae = torch.cat([to_display(img) for img in vae_imgs], dim=0)
            row_spnn = torch.cat([to_display(img) for img in spnn_imgs], dim=0)
            grid = torch.cat([row_vae, row_spnn], dim=0)
            grid_path = os.path.join(test_sample_dir, f"test_cycles_{img_idx:03d}.png")
            save_image(grid, grid_path, nrow=args.num_cycles + 1, padding=4, pad_value=1.0)
            wandb.log({"test/cycle_grids": wandb.Image(grid_path,
                       caption=f"Image {img_idx} — Row1: VAE, Row2: SPNN")})

    # Average Penrose
    avg_penrose = {}
    for key in all_penrose[0]:
        avg_penrose[key] = sum(p[key] for p in all_penrose) / len(all_penrose)
    print("Penrose diagnostics (averaged):")
    print_penrose_metrics(avg_penrose)
    wandb.log({"test/" + k: v for k, v in avg_penrose.items()})

    # Average cycle metrics
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

    num_tested = min(args.num_test_images, len(dataset)) if args.num_test_images > 0 else len(dataset)
    print(f"\nAveraged over {num_tested} test images.")


def parse_args():
    p = argparse.ArgumentParser(
        description="CelebA-HQ SPNN cycle-consistency test")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--num_cycles", type=int, default=5)
    p.add_argument("--num_test_images", type=int, default=0,
                   help="Limit test images (0=all)")
    p.add_argument("--num_save_images", type=int, default=30)
    p.add_argument("--sample_dir", type=str,
                   default="celebahq_experiment/samples")
    p.add_argument("--mix_type", type=str, default="cayley")
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--scale_bound", type=float, default=2.0)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    wandb.init(project="spnn-celebahq", config=vars(args))
    run_test(args)
    wandb.finish()
