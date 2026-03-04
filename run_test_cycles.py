import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import math
from tqdm import tqdm
from diffusers import AutoencoderKL
import wandb
from models import SPNNAutoencoder
from dataset import CelebAHQDataset
from diagnostics import penrose_check, print_penrose_metrics

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def to_display(tensor):
    return ((tensor.cpu() + 1) / 2).clamp(0, 1)


def calc_psnr(a, b):
    m = F.mse_loss(a, b).item()
    return 10 * math.log10(4.0 / m) if m > 0 else float("inf")


def calc_mse(a, b):
    return F.mse_loss(a, b).item()


# ──────────────────────── Load models ────────────────────────

def load_sd_vae():
    print("Loading SD-VAE from timbrooks/instruct-pix2pix...")
    vae = AutoencoderKL.from_pretrained(
        "timbrooks/instruct-pix2pix", subfolder="vae"
    )
    vae.eval().to(DEVICE)
    for p in vae.parameters():
        p.requires_grad = False
    return vae


def load_spnn(ckpt_path, mix_type, hidden, scale_bound):
    print(f"Loading SPNN from {ckpt_path}...")
    spnn = SPNNAutoencoder(mix_type=mix_type, hidden=hidden, scale_bound=scale_bound)
    state = torch.load(ckpt_path, map_location=DEVICE)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    spnn.load_state_dict(state)
    spnn.eval().to(DEVICE)
    return spnn


# ──────────────────────── Single cycle functions ────────────────────────

@torch.no_grad()
def vae_cycle(vae, x):
    """SD-VAE encode -> decode. Lossy each pass."""
    posterior = vae.encode(x).latent_dist
    z = posterior.mode()
    return vae.decode(z).sample


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

    vae = load_sd_vae()
    spnn = load_spnn(args.checkpoint, args.mix_type, args.hidden, args.scale_bound)

    dataset = CelebAHQDataset(
        img_size=args.img_size, split="test", test_ratio=args.test_ratio,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    print(f"Test set: {len(dataset)} images\n")

    wandb.init(
        project=args.wandb_project, entity=args.wandb_entity, job_type="test",
        config=vars(args),
    )

    test_sample_dir = os.path.join(args.sample_dir, "test")
    os.makedirs(test_sample_dir, exist_ok=True)

    # ── Aggregate metrics per cycle ──
    cycle_metrics = {c: [] for c in range(1, args.num_cycles + 1)}
    all_penrose = []

    for img_idx, original in enumerate(tqdm(loader, desc="Testing")):
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
            wandb.log({"test/cycle_grids": wandb.Image(grid_path, caption=f"Image {img_idx} — Row1: VAE, Row2: SPNN (pinv)")})

            # ── Summary: original | VAE@last | SPNN@last ──
            summary = torch.cat([
                to_display(original),
                to_display(vae_imgs[-1]),
                to_display(spnn_imgs[-1]),
            ], dim=0)
            summary_path = os.path.join(test_sample_dir, f"test_summary_{img_idx:03d}.png")
            save_image(summary, summary_path, nrow=3, padding=4, pad_value=1.0)
            wandb.log({"test/summaries": wandb.Image(summary_path, caption=f"Image {img_idx} — Orig | VAE@{args.num_cycles} | SPNN@{args.num_cycles}")})

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

    wandb.finish()