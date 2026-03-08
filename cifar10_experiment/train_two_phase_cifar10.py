"""
Two-phase SPNN training for diffusion-compatible encoding on CIFAR-10.

Phase 1 — Encoder alignment (train s, t, mix; freeze r):
    Push SPNN.encode(x) to match VAE.encode(x) and make the UNet behave
    identically on SPNN latents vs VAE latents. No decoder loss — no
    conflicting gradients.

    Losses:
      - align:     MSE(SPNN.encode(x), VAE.encode(x))
      - diffusion: MSE(UNet(noisy_z_spnn), UNet(noisy_z_vae))  [behavior matching]

Phase 2 — Decoder training (train r only; freeze s, t, mix):
    With encoder fixed, train r to reconstruct the discarded channels.
    r's task is now well-defined: given y from the fixed encoder, estimate x1.

    Losses:
      - decoder:   MSE(SPNN.decode(z_vae), VAE.decode(z_vae))
      - roundtrip: MSE(SPNN.decode(SPNN.encode(x)), x)
      - lpips:     perceptual similarity (optional)
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
    get_cifar10_loaders, get_vae_pairs, load_simple_vae, save_comparison,
    LatentDiscriminator)
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
# Param control helpers
# ═══════════════════════════════════════════════════════════

def set_requires_grad(spnn, s_t_mix=True, r=True):
    """Control which SPNN sub-networks are trainable."""
    for block in spnn.blocks:
        if isinstance(block, ConvPINNBlock):
            for p in block.r.parameters():
                p.requires_grad = r
            for name in ['s', 't', 'mix']:
                for p in getattr(block, name).parameters():
                    p.requires_grad = s_t_mix


def count_params(spnn):
    """Count trainable and frozen params."""
    trainable = sum(p.numel() for p in spnn.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in spnn.parameters() if not p.requires_grad)
    return trainable, frozen


def get_trainable_params(spnn):
    return [p for p in spnn.parameters() if p.requires_grad]


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
# Losses
# ═══════════════════════════════════════════════════════════

def diffusion_behavior_loss(ldm, z_spnn, z_vae, class_labels):
    """
    Match UNet's behavior on SPNN latents vs VAE latents.
    Same noise ε, same timestep t. Gradient flows through z_spnn only.
    """
    sampler = ldm.sampler
    network = ldm.network

    B = z_spnn.shape[0]
    eps = torch.randn_like(z_spnn)
    t = torch.randint(0, sampler.T, (B,), device=z_spnn.device)

    alpha_bar_t = sampler.alpha_bar[t].view(B, 1, 1, 1).to(z_spnn.device)

    z_t_spnn = alpha_bar_t.sqrt() * z_spnn + (1 - alpha_bar_t).sqrt() * eps
    z_t_vae = alpha_bar_t.sqrt() * z_vae + (1 - alpha_bar_t).sqrt() * eps

    eps_pred_spnn = network(x=z_t_spnn, t=t, y=class_labels)
    with torch.no_grad():
        eps_pred_vae = network(x=z_t_vae, t=t, y=class_labels)

    return F.mse_loss(eps_pred_spnn, eps_pred_vae)


# ═══════════════════════════════════════════════════════════
# Eval helpers
# ═══════════════════════════════════════════════════════════

@torch.no_grad()
def eval_metrics(spnn, vae, penrose_images, penrose_latent):
    """Compute Penrose + alignment metrics."""
    spnn.eval()
    p_metrics = penrose_check(spnn, penrose_images, penrose_latent, DEVICE)
    print_penrose_metrics(p_metrics)

    z_spnn = spnn.encode(penrose_images)
    lat_mse = F.mse_loss(z_spnn, penrose_latent).item()
    cos_sim = F.cosine_similarity(
        z_spnn.view(z_spnn.size(0), -1),
        penrose_latent.view(penrose_latent.size(0), -1), dim=1
    ).mean().item()
    print(f"  Latent align: MSE={lat_mse:.6f}, cosine={cos_sim:.6f}")

    # Decoder quality: decode VAE latent, compare to VAE decode
    spnn_dec = spnn.decode(penrose_latent[:8])
    vae_dec = vae.decode(penrose_latent[:8])
    dec_mse = F.mse_loss(spnn_dec, vae_dec).item()
    print(f"  Decoder MSE(SPNN vs VAE): {dec_mse:.6f}")

    extra = {"latent_mse": lat_mse, "cosine_sim": cos_sim, "decoder_mse": dec_mse}
    return {**p_metrics, **extra}


# ═══════════════════════════════════════════════════════════
# Phase 1: Encoder alignment
# ═══════════════════════════════════════════════════════════

def phase1_encoder(args, spnn, ldm, vae, train_loader, penrose_images, penrose_latent):
    """Train s, t, mix to align encoder with VAE latent space."""
    print(f"\n{'='*60}")
    print(f"PHASE 1: Encoder Alignment")
    print(f"{'='*60}")

    # Freeze r, train s/t/mix
    set_requires_grad(spnn, s_t_mix=True, r=False)
    trainable, frozen = count_params(spnn)
    print(f"  Trainable: {trainable:,} (s, t, mix)")
    print(f"  Frozen:    {frozen:,} (r)")

    params = get_trainable_params(spnn)
    optimizer = torch.optim.AdamW(params, lr=args.lr_phase1, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs_phase1 * len(train_loader), eta_min=1e-7)
    mse_loss = nn.MSELoss()

    # ── Adversarial: latent discriminator ──
    use_adv = args.lambda_adv_p1 > 0
    disc = None
    disc_optimizer = None
    if use_adv:
        # Infer latent shape from a test encode
        with torch.no_grad():
            test_z = vae.encode(penrose_images[:1]).mode()
        in_ch, spatial = test_z.shape[1], test_z.shape[2]
        disc = LatentDiscriminator(in_ch=in_ch, spatial=spatial).to(DEVICE)
        disc_optimizer = torch.optim.Adam(disc.parameters(), lr=args.lr_disc,
                                          betas=(0.0, 0.9))
        d_params = sum(p.numel() for p in disc.parameters())
        print(f"  Discriminator: {d_params:,} params (latent {in_ch}x{spatial}x{spatial})")

    # Baseline
    print("\nBaseline (before phase 1):")
    metrics = eval_metrics(spnn, vae, penrose_images, penrose_latent)
    wandb.log({f"p1/{k}": v for k, v in metrics.items()}, step=0)

    for epoch in range(1, args.epochs_phase1 + 1):
        spnn.train()
        if disc is not None:
            disc.train()
        epoch_loss = 0.0

        pbar = tqdm(train_loader, desc=f"P1 Epoch {epoch}/{args.epochs_phase1}")
        for images, labels in pbar:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            with torch.no_grad():
                vae_latent = vae.encode(images).mode()

            z_spnn = spnn.encode(images)

            # ── (a) Train discriminator ──
            d_loss = torch.tensor(0.0, device=DEVICE)
            if use_adv:
                disc_optimizer.zero_grad()
                # Real = VAE latents, Fake = SPNN latents (detached)
                d_real = disc(vae_latent)
                d_fake = disc(z_spnn.detach())
                # Hinge loss (more stable than BCE for latent GANs)
                d_loss = (F.relu(1.0 - d_real).mean() + F.relu(1.0 + d_fake).mean())
                d_loss.backward()
                disc_optimizer.step()

            # ── (b) Train encoder (s, t, mix) ──
            # Align loss: direct latent matching
            align_loss = mse_loss(z_spnn, vae_latent)

            # Diffusion behavior loss: UNet acts same on both
            diff_loss = diffusion_behavior_loss(ldm, z_spnn, vae_latent, labels)

            # Adversarial generator loss: fool D into thinking SPNN latents are VAE
            g_loss = torch.tensor(0.0, device=DEVICE)
            if use_adv:
                g_loss = -disc(z_spnn).mean()  # non-saturating hinge generator

            loss = (args.lambda_align_p1 * align_loss
                    + args.lambda_diff_p1 * diff_loss
                    + args.lambda_adv_p1 * g_loss)

            optimizer.zero_grad()
            loss.backward()
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(params, max_norm=args.max_grad_norm)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            log_dict = {
                "p1/loss": loss.item(),
                "p1/align_loss": align_loss.item(),
                "p1/diff_loss": diff_loss.item(),
                "p1/lr": scheduler.get_last_lr()[0],
            }
            if use_adv:
                log_dict["p1/d_loss"] = d_loss.item()
                log_dict["p1/g_loss"] = g_loss.item()
            wandb.log(log_dict)
            pbar.set_postfix(loss=f"{loss.item():.4f}", align=f"{align_loss.item():.4f}",
                             g=f"{g_loss.item():.2f}" if use_adv else "off")

        avg = epoch_loss / len(train_loader)
        print(f"  P1 Epoch {epoch} — avg loss: {avg:.6f}")

        if epoch % args.save_every == 0 or epoch == args.epochs_phase1:
            metrics = eval_metrics(spnn, vae, penrose_images, penrose_latent)
            wandb.log({f"p1/{k}": v for k, v in metrics.items()})

            ckpt_path = os.path.join(args.output_dir, f"spnn_p1_epoch{epoch:03d}.pt")
            save_dict = {
                "epoch": epoch,
                "phase": 1,
                "model_state_dict": spnn.state_dict(),
                "stages": CIFAR10_STAGES,
            }
            if disc is not None:
                save_dict["disc_state_dict"] = disc.state_dict()
            torch.save(save_dict, ckpt_path)
            print(f"  Saved: {ckpt_path}")

    # Save phase 1 final
    p1_path = os.path.join(args.output_dir, "spnn_phase1_final.pt")
    save_dict = {
        "model_state_dict": spnn.state_dict(),
        "stages": CIFAR10_STAGES,
        "phase": 1,
    }
    if disc is not None:
        save_dict["disc_state_dict"] = disc.state_dict()
    torch.save(save_dict, p1_path)
    print(f"\nPhase 1 complete. Saved: {p1_path}")
    return spnn


# ═══════════════════════════════════════════════════════════
# Phase 2: Decoder training
# ═══════════════════════════════════════════════════════════

def phase2_decoder(args, spnn, vae, train_loader, penrose_images, penrose_latent):
    """Train r only to match decoder quality with frozen s, t, mix."""
    print(f"\n{'='*60}")
    print(f"PHASE 2: Decoder Training (r only)")
    print(f"{'='*60}")

    # Freeze s/t/mix, train r
    set_requires_grad(spnn, s_t_mix=False, r=True)
    trainable, frozen = count_params(spnn)
    print(f"  Trainable: {trainable:,} (r)")
    print(f"  Frozen:    {frozen:,} (s, t, mix)")

    params = get_trainable_params(spnn)
    optimizer = torch.optim.AdamW(params, lr=args.lr_phase2, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs_phase2 * len(train_loader), eta_min=1e-7)
    mse_loss = nn.MSELoss()

    # LPIPS
    lpips_fn = None
    if args.lambda_lpips_p2 > 0:
        import lpips
        lpips_fn = lpips.LPIPS(net="vgg").to(DEVICE)
        lpips_fn.eval()
        for p in lpips_fn.parameters():
            p.requires_grad = False
        print("  LPIPS enabled (upsample to 64x64)")

    # Baseline
    print("\nBaseline (before phase 2):")
    metrics = eval_metrics(spnn, vae, penrose_images, penrose_latent)
    wandb.log({f"p2/{k}": v for k, v in metrics.items()})

    for epoch in range(1, args.epochs_phase2 + 1):
        spnn.train()
        epoch_loss = 0.0

        pbar = tqdm(train_loader, desc=f"P2 Epoch {epoch}/{args.epochs_phase2}")
        for images, _labels in pbar:
            images = images.to(DEVICE)

            with torch.no_grad():
                vae_latent, vae_decoded = get_vae_pairs(vae, images)

            # Decoder loss: SPNN.decode(z_vae) ≈ VAE.decode(z_vae)
            spnn_decoded = spnn.decode(vae_latent)
            decoder_loss = mse_loss(spnn_decoded, vae_decoded)

            # Roundtrip loss: decode(encode(x)) ≈ x
            roundtrip_loss = torch.tensor(0.0, device=DEVICE)
            if args.lambda_roundtrip_p2 > 0:
                spnn_latent = spnn.encode(images)  # uses frozen s,t,mix
                spnn_recon = spnn.decode(spnn_latent)
                roundtrip_loss = mse_loss(spnn_recon, images)

            # LPIPS
            lpips_loss = torch.tensor(0.0, device=DEVICE)
            if lpips_fn is not None:
                up_spnn = F.interpolate(spnn_decoded, size=64, mode='bilinear',
                                        align_corners=False)
                up_vae = F.interpolate(vae_decoded, size=64, mode='bilinear',
                                       align_corners=False)
                lpips_loss = lpips_fn(up_spnn, up_vae).mean()

            loss = (decoder_loss
                    + args.lambda_roundtrip_p2 * roundtrip_loss
                    + args.lambda_lpips_p2 * lpips_loss)

            optimizer.zero_grad()
            loss.backward()
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(params, max_norm=args.max_grad_norm)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            wandb.log({
                "p2/loss": loss.item(),
                "p2/decoder_loss": decoder_loss.item(),
                "p2/roundtrip_loss": roundtrip_loss.item(),
                "p2/lpips_loss": lpips_loss.item(),
                "p2/lr": scheduler.get_last_lr()[0],
            })
            pbar.set_postfix(loss=f"{loss.item():.4f}", dec=f"{decoder_loss.item():.4f}")

        avg = epoch_loss / len(train_loader)
        print(f"  P2 Epoch {epoch} — avg loss: {avg:.6f}")

        if epoch % args.save_every == 0 or epoch == args.epochs_phase2:
            metrics = eval_metrics(spnn, vae, penrose_images, penrose_latent)
            wandb.log({f"p2/{k}": v for k, v in metrics.items()})

            # Save comparison images
            with torch.no_grad():
                sample_decoded = spnn.decode(penrose_latent[:4])
                _, vae_dec = get_vae_pairs(vae, penrose_images[:4])
            save_comparison(sample_decoded, vae_dec, penrose_images[:4],
                            epoch, 0, os.path.join(args.output_dir, "samples_p2"))

            ckpt_path = os.path.join(args.output_dir, f"spnn_p2_epoch{epoch:03d}.pt")
            torch.save({
                "epoch": epoch,
                "phase": 2,
                "model_state_dict": spnn.state_dict(),
                "stages": CIFAR10_STAGES,
            }, ckpt_path)
            print(f"  Saved: {ckpt_path}")

    # Save final
    final_path = os.path.join(args.output_dir, "spnn_two_phase_final.pt")
    torch.save({
        "model_state_dict": spnn.state_dict(),
        "stages": CIFAR10_STAGES,
        "phase": "complete",
    }, final_path)
    print(f"\nPhase 2 complete. Final model: {final_path}")
    return spnn


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "samples_p2"), exist_ok=True)

    train_loader, test_loader = get_cifar10_loaders(args.batch_size, args.num_workers)
    print(f"CIFAR-10: {len(train_loader.dataset)} train, {len(test_loader.dataset)} test")

    # Load frozen LDM + VAE
    print("Loading frozen LDM...")
    ldm, vae = load_ldm_frozen(args.ldm_path)

    # Build fresh SPNN (random init)
    print("Building SPNN (random init)...")
    spnn = SPNNAutoencoderConfigurable(
        stages=CIFAR10_STAGES,
        mix_type=args.mix_type,
        hidden=args.hidden,
        scale_bound=args.scale_bound,
    ).to(DEVICE)
    total = sum(p.numel() for p in spnn.parameters())
    print(f"  Total params: {total:,}")

    # Fixed test batch for eval
    test_iter = iter(test_loader)
    penrose_images, _ = next(test_iter)
    penrose_images = penrose_images[:args.penrose_batch_size].to(DEVICE)
    penrose_latent, _ = get_vae_pairs(vae, penrose_images)
    del test_iter

    wandb.init(project="spnn-cifar10-twophase", config=vars(args))

    # Phase 1: Encoder alignment
    spnn = phase1_encoder(args, spnn, ldm, vae, train_loader,
                          penrose_images, penrose_latent)

    # Phase 2: Decoder training
    spnn = phase2_decoder(args, spnn, vae, train_loader,
                          penrose_images, penrose_latent)

    wandb.finish()
    print(f"\nDone. Results in {args.output_dir}/")


def parse_args():
    p = argparse.ArgumentParser(
        description="Two-phase SPNN training: encoder alignment then decoder training")

    p.add_argument("--ldm_path", type=str,
                   default=os.path.join(SLDM_ROOT, 'models', 'cifar_ldm.pth'))

    # Phase 1: encoder
    p.add_argument("--epochs_phase1", type=int, default=50)
    p.add_argument("--lr_phase1", type=float, default=3e-4)
    p.add_argument("--lambda_align_p1", type=float, default=1.0,
                   help="Latent MSE alignment weight (phase 1)")
    p.add_argument("--lambda_diff_p1", type=float, default=1.0,
                   help="Diffusion behavior matching weight (phase 1)")
    p.add_argument("--lambda_adv_p1", type=float, default=0.1,
                   help="Adversarial latent matching weight (phase 1, 0=disabled)")
    p.add_argument("--lr_disc", type=float, default=1e-4,
                   help="Discriminator learning rate")

    # Phase 2: decoder
    p.add_argument("--epochs_phase2", type=int, default=50)
    p.add_argument("--lr_phase2", type=float, default=3e-4)
    p.add_argument("--lambda_roundtrip_p2", type=float, default=1.0,
                   help="Roundtrip loss weight (phase 2)")
    p.add_argument("--lambda_lpips_p2", type=float, default=0.1,
                   help="LPIPS perceptual loss weight (phase 2)")

    # Shared
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--max_grad_norm", type=float, default=2.0)
    p.add_argument("--save_every", type=int, default=5)
    p.add_argument("--penrose_batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)

    # Model
    p.add_argument("--mix_type", type=str, default="cayley")
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--scale_bound", type=float, default=2.0)

    p.add_argument("--output_dir", type=str,
                   default="cifar10_experiment/checkpoints_twophase")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
