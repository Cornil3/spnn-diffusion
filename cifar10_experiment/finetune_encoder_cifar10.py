"""
Fine-tune SPNN encoder for diffusion compatibility on CIFAR-10.

Freezes the r networks (decode-only) and fine-tunes s, t, mix so that
SPNN.encode(x) produces latents the pretrained UNet can denoise well.

Core loss:
    z = SPNN.encode(x)
    z_t = sqrt(α_t) * z + sqrt(1-α_t) * ε
    ε_pred = UNet(z_t, t)
    diffusion_loss = MSE(ε_pred, ε)

If the UNet can predict noise on SPNN-encoded latents, they're in-distribution.

Regularization losses (from original training) keep decoder quality:
    - decoder_loss:  SPNN.decode(z_vae) ≈ VAE.decode(z_vae)
    - cycle_loss:    SPNN.encode(SPNN.decode(z)) ≈ z
    - align_loss:    SPNN.encode(x) ≈ VAE.encode(x)
"""

import argparse
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm
import wandb

# ── SPNN ──
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from cifar10_experiment.train_cifar10 import (
    SPNNAutoencoderConfigurable, CIFAR10_STAGES, _load_checkpoint,
    get_cifar10_loaders, get_vae_pairs, load_simple_vae, save_comparison)
from diagnostics import penrose_check, print_penrose_metrics
from models import ConvPINNBlock

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
# Freeze / unfreeze helpers
# ═══════════════════════════════════════════════════════════

def freeze_r_networks(spnn):
    """Freeze r networks in all ConvPINNBlocks — decode quality preserved."""
    frozen = 0
    trainable = 0
    for block in spnn.blocks:
        if isinstance(block, ConvPINNBlock):
            for p in block.r.parameters():
                p.requires_grad = False
                frozen += p.numel()
            for name in ['s', 't', 'mix']:
                for p in getattr(block, name).parameters():
                    p.requires_grad = True
                    trainable += p.numel()
    print(f"  Frozen (r networks): {frozen:,} params")
    print(f"  Trainable (s, t, mix): {trainable:,} params")


def load_ldm_frozen(ldm_path):
    """Load trained LDM (UNet + DDIM + VAE) with all params frozen."""
    vae = VariationalAutoEncoder(CONFIG_PATH)
    vae = _load_checkpoint(vae, os.path.join(SLDM_ROOT, 'models', 'cifar_vae.pth'))
    vae.eval().to(DEVICE)
    for p in vae.parameters():
        p.requires_grad = False

    sampler = DDIM(CONFIG_PATH)
    cond_encoder = ClassEncoder(CONFIG_PATH)
    network = UnetWrapper(Unet, CONFIG_PATH, cond_encoder)
    ldm = LatentDiffusionModel(network, sampler, vae)
    ldm = _load_checkpoint(ldm, ldm_path)
    ldm.eval().to(DEVICE)
    for p in ldm.parameters():
        p.requires_grad = False

    return ldm, vae


# ═══════════════════════════════════════════════════════════
# Diffusion loss
# ═══════════════════════════════════════════════════════════

def diffusion_behavior_loss(ldm, z_spnn, z_vae, class_labels):
    """
    Match UNet's behavior on SPNN latents vs VAE latents.

    Same noise ε, same timestep t applied to both:
        z_t_spnn = q_sample(z_spnn, t, ε)
        z_t_vae  = q_sample(z_vae,  t, ε)
        loss = MSE(UNet(z_t_spnn, t), UNet(z_t_vae, t))

    Gradient flows through z_spnn only (z_vae is detached target).
    No adversarial-like gradients — we're matching behavior, not predicting noise.
    """
    sampler = ldm.sampler
    network = ldm.network

    B = z_spnn.shape[0]
    eps = torch.randn_like(z_spnn)
    t = torch.randint(0, sampler.T, (B,), device=z_spnn.device)

    # Extract alpha_bar for timestep t
    alpha_bar_t = sampler.alpha_bar[t].view(B, 1, 1, 1).to(z_spnn.device)

    # Forward diffusion with same noise on both
    z_t_spnn = alpha_bar_t.sqrt() * z_spnn + (1 - alpha_bar_t).sqrt() * eps
    z_t_vae = alpha_bar_t.sqrt() * z_vae + (1 - alpha_bar_t).sqrt() * eps

    # UNet predictions — grad flows through z_t_spnn only
    eps_pred_spnn = network(x=z_t_spnn, t=t, y=class_labels)
    with torch.no_grad():
        eps_pred_vae = network(x=z_t_vae, t=t, y=class_labels)

    # Match UNet behavior: if equal, SPNN latents are indistinguishable from VAE
    loss = F.mse_loss(eps_pred_spnn, eps_pred_vae)
    return loss


# ═══════════════════════════════════════════════════════════
# Training loop
# ═══════════════════════════════════════════════════════════

def finetune(args):
    print(f"Device: {DEVICE}")
    print(f"Fine-tuning from: {args.checkpoint}")

    os.makedirs(args.output_dir, exist_ok=True)
    sample_dir = os.path.join(args.output_dir, "samples")
    os.makedirs(sample_dir, exist_ok=True)

    # ── Dataset ──
    train_loader, test_loader = get_cifar10_loaders(
        args.batch_size, args.num_workers)
    print(f"CIFAR-10: {len(train_loader.dataset)} train, "
          f"{len(test_loader.dataset)} test")

    # ── Load frozen LDM + VAE ──
    print("Loading frozen LDM...")
    ldm, vae = load_ldm_frozen(args.ldm_path)

    # ── Load pretrained SPNN ──
    print(f"Loading SPNN from {args.checkpoint}...")
    spnn = SPNNAutoencoderConfigurable(
        stages=CIFAR10_STAGES,
        mix_type=args.mix_type,
        hidden=args.hidden,
        scale_bound=args.scale_bound,
    ).to(DEVICE)
    state = torch.load(args.checkpoint, map_location=DEVICE, weights_only=True)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    spnn.load_state_dict(state)

    # ── Freeze r, train s/t/mix ──
    print("Freezing r networks...")
    freeze_r_networks(spnn)

    # ── Fixed test batch for Penrose checks ──
    test_iter = iter(test_loader)
    penrose_images, _ = next(test_iter)
    penrose_images = penrose_images[:args.penrose_batch_size].to(DEVICE)
    penrose_latent, _ = get_vae_pairs(vae, penrose_images)
    del test_iter

    # ── Optimizer (only trainable params) ──
    trainable_params = [p for p in spnn.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_epochs * len(train_loader), eta_min=1e-7)
    mse_loss = nn.MSELoss()

    # ── WandB ──
    wandb.init(project="spnn-cifar10-finetune", config=vars(args))

    # ── Baseline metrics (epoch 0, before any training) ──
    print("Baseline metrics (before fine-tuning):")
    p_metrics = penrose_check(spnn, penrose_images, penrose_latent, DEVICE)
    print_penrose_metrics(p_metrics)
    with torch.no_grad():
        z_spnn_test = spnn.encode(penrose_images)
        lat_mse = F.mse_loss(z_spnn_test, penrose_latent).item()
        cos_sim = F.cosine_similarity(
            z_spnn_test.view(z_spnn_test.size(0), -1),
            penrose_latent.view(penrose_latent.size(0), -1), dim=1
        ).mean().item()
    print(f"  Latent align: MSE={lat_mse:.6f}, cosine={cos_sim:.6f}")
    wandb.log({**p_metrics, "ft/latent_mse": lat_mse, "ft/cosine_sim": cos_sim, "epoch": 0})

    for epoch in range(1, args.num_epochs + 1):
        spnn.train()
        epoch_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.num_epochs}")
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            # ── VAE targets ──
            with torch.no_grad():
                vae_latent, vae_decoded = get_vae_pairs(vae, images)

            # ── Diffusion behavior loss: UNet behaves same on SPNN vs VAE latents ──
            z_spnn = spnn.encode(images)
            diff_loss = diffusion_behavior_loss(ldm, z_spnn, vae_latent, labels)

            # ── Decoder loss: SPNN.decode(z_vae) ≈ VAE.decode(z_vae) ──
            dec_loss = torch.tensor(0.0, device=DEVICE)
            if args.lambda_decoder > 0:
                spnn_decoded = spnn.decode(vae_latent)
                dec_loss = mse_loss(spnn_decoded, vae_decoded)

            # ── Cycle loss: encode(decode(z)) ≈ z ──
            cycle_loss = torch.tensor(0.0, device=DEVICE)
            if args.lambda_cycle > 0:
                if args.lambda_decoder == 0:
                    spnn_decoded = spnn.decode(vae_latent)
                re_encoded = spnn.encode(spnn_decoded)
                cycle_loss = mse_loss(re_encoded, vae_latent)

            # ── Align loss: SPNN.encode(x) ≈ VAE.encode(x) ──
            align_loss = torch.tensor(0.0, device=DEVICE)
            if args.lambda_align > 0:
                align_loss = mse_loss(z_spnn, vae_latent)

            loss = (args.lambda_diffusion * diff_loss
                    + args.lambda_decoder * dec_loss
                    + args.lambda_cycle * cycle_loss
                    + args.lambda_align * align_loss)

            optimizer.zero_grad()
            loss.backward()
            if args.max_grad_norm > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    trainable_params, max_norm=args.max_grad_norm)
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    trainable_params, max_norm=float('inf'))
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()

            wandb.log({
                "ft/loss": loss.item(),
                "ft/diffusion_loss": diff_loss.item(),
                "ft/decoder_loss": dec_loss.item(),
                "ft/cycle_loss": cycle_loss.item(),
                "ft/align_loss": align_loss.item(),
                "ft/lr": scheduler.get_last_lr()[0],
                "ft/grad_norm": grad_norm.item(),
            })

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "diff": f"{diff_loss.item():.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}",
            })

        avg_loss = epoch_loss / len(train_loader)
        wandb.log({"ft/epoch_avg_loss": avg_loss, "epoch": epoch})
        print(f"  Epoch {epoch} — avg loss: {avg_loss:.6f}")

        # ── Penrose diagnostics + checkpoint ──
        if epoch % args.save_every == 0:
            p_metrics = penrose_check(spnn, penrose_images, penrose_latent, DEVICE)
            print_penrose_metrics(p_metrics)
            wandb.log({**p_metrics, "epoch": epoch})

            # Latent alignment on penrose batch
            with torch.no_grad():
                z_spnn_test = spnn.encode(penrose_images)
                lat_mse = F.mse_loss(z_spnn_test, penrose_latent).item()
                cos_sim = F.cosine_similarity(
                    z_spnn_test.view(z_spnn_test.size(0), -1),
                    penrose_latent.view(penrose_latent.size(0), -1), dim=1
                ).mean().item()
            print(f"  Latent align: MSE={lat_mse:.6f}, cosine={cos_sim:.6f}")
            wandb.log({"ft/latent_mse": lat_mse, "ft/cosine_sim": cos_sim, "epoch": epoch})

            # Save comparison images
            with torch.no_grad():
                sample_decoded = spnn.decode(penrose_latent[:4])
                _, vae_dec = get_vae_pairs(vae, penrose_images[:4])
            save_comparison(sample_decoded, vae_dec, penrose_images[:4],
                            epoch, 0, sample_dir)

            spnn.train()

            ckpt_path = os.path.join(args.output_dir,
                                     f"spnn_cifar10_ft_epoch{epoch:03d}.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": spnn.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
                "stages": CIFAR10_STAGES,
                "finetuned_from": args.checkpoint,
            }, ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}")

    # ── Final save ──
    final_path = os.path.join(args.output_dir, "spnn_cifar10_ft_final.pt")
    torch.save({
        "model_state_dict": spnn.state_dict(),
        "stages": CIFAR10_STAGES,
        "finetuned_from": args.checkpoint,
    }, final_path)
    print(f"\nFine-tuning complete. Final model: {final_path}")
    wandb.finish()


def parse_args():
    p = argparse.ArgumentParser(
        description="Fine-tune SPNN encoder for diffusion compatibility (CIFAR-10)")
    # Checkpoint
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Pretrained SPNN checkpoint to fine-tune from")
    p.add_argument("--ldm_path", type=str,
                   default=os.path.join(SLDM_ROOT, 'models', 'cifar_ldm.pth'),
                   help="Trained LDM checkpoint path")
    # Training
    p.add_argument("--num_epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-5,
                   help="Small LR for fine-tuning")
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    # Loss weights
    p.add_argument("--lambda_diffusion", type=float, default=1.0,
                   help="Weight of diffusion loss (UNet can denoise SPNN latents)")
    p.add_argument("--lambda_decoder", type=float, default=0.5,
                   help="Weight of decoder preservation loss")
    p.add_argument("--lambda_cycle", type=float, default=0.5,
                   help="Weight of cycle consistency loss")
    p.add_argument("--lambda_align", type=float, default=0.1,
                   help="Weight of direct latent alignment loss")
    # Model
    p.add_argument("--mix_type", type=str, default="cayley")
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--scale_bound", type=float, default=2.0)
    # Misc
    p.add_argument("--save_every", type=int, default=2)
    p.add_argument("--penrose_batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--output_dir", type=str,
                   default="cifar10_experiment/checkpoints_ft")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    finetune(args)
