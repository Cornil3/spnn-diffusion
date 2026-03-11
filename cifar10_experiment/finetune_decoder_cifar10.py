"""
Fine-tune SPNN decoder on sampled VAE latents (not just the mode).

Pipeline per batch:
    x -> VAE.encode(x) -> posterior(μ, σ)
    z = μ + α·σ·ε        (stochastic sample, α controls spread)
    x_hat = SPNN.decode(z)

Losses:
    distill  = MSE(SPNN.decode(z), VAE.decode(z))   # match VAE decoder
    recon    = MSE(SPNN.decode(z), x)                # match original image
    lpips    = LPIPS(SPNN.decode(z), x)              # perceptual
    roundtrip = MSE(SPNN.decode(SPNN.encode(x)), x)  # keep encoder working

This teaches the SPNN decoder to handle the full latent distribution
(not just μ), making it robust to UNet-denoised latents during inference.
"""

import argparse
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
from tqdm import tqdm
import wandb
from accelerate import Accelerator

# ── SPNN ──
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from models import ConvPINNBlock

# ── Reuse from train_cifar10 ──
from cifar10_experiment.train_cifar10 import (
    SPNNAutoencoderConfigurable, CIFAR10_STAGES,
    load_simple_vae_to, get_cifar10_loaders,
)

# ── Simple LDM ──
SLDM_ROOT = os.path.join(os.path.dirname(__file__), '..',
    'simple-latent-diffusion-model-master', 'simple-latent-diffusion-model')
sys.path.insert(0, SLDM_ROOT)


# ═══════════════════════════════════════════════════════════
# Decoder wrapper for DDP
# ═══════════════════════════════════════════════════════════

class DecoderWrapper(nn.Module):
    """
    Wraps the SPNN so that forward() = decode().
    This lets Accelerate/DDP sync gradients on the r networks
    (the only trainable params) through the standard forward() path.
    """
    def __init__(self, spnn):
        super().__init__()
        self.spnn = spnn

    def forward(self, z):
        return self.spnn.decode(z)


def save_grid(spnn_decoded, vae_decoded, original, path):
    """Save comparison grid: original | VAE decode | SPNN decode."""
    n = min(4, original.size(0))
    orig = (original[:n].cpu() + 1) / 2
    vae_r = (vae_decoded[:n].cpu() + 1) / 2
    spnn_r = (spnn_decoded[:n].detach().cpu() + 1) / 2
    grid = torch.cat([orig, vae_r, spnn_r], dim=0)
    save_image(grid, path, nrow=n, padding=2)


def train(args):
    accelerator = Accelerator(mixed_precision='fp16')
    device = accelerator.device
    is_main = accelerator.is_main_process

    if is_main:
        print(f"Device: {device}  |  Num processes: {accelerator.num_processes}")

    run_name = (f"ft_dec_a{args.alpha}_dist{args.lambda_distill}"
                f"_rec{args.lambda_recon}_lp{args.lambda_lpips}"
                f"_rt{args.lambda_roundtrip}_h{args.hidden}")
    args.output_dir = os.path.join(args.output_dir, run_name)

    sample_dir = os.path.join(args.output_dir, "samples")
    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(sample_dir, exist_ok=True)
        print(f"Output dir: {args.output_dir}")

    # ── Dataset ──
    train_loader, test_loader = get_cifar10_loaders(
        args.batch_size, args.num_workers)
    if is_main:
        print(f"CIFAR-10: {len(train_loader.dataset)} train, "
              f"{len(test_loader.dataset)} test")

    # ── Frozen VAE ──
    vae = load_simple_vae_to(device)

    # ── SPNN (load pretrained checkpoint) ──
    spnn = SPNNAutoencoderConfigurable(
        stages=CIFAR10_STAGES,
        mix_type=args.mix_type,
        hidden=args.hidden,
        scale_bound=args.scale_bound,
    ).to(device)

    assert args.checkpoint is not None, "--checkpoint is required"
    if is_main:
        print(f"Loading SPNN from {args.checkpoint}...")
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    spnn.load_state_dict(state)

    total_params = sum(p.numel() for p in spnn.parameters())
    if is_main:
        print(f"SPNN total params: {total_params:,}")

    # ── Freeze encoder (s, t, mix) — only train decoder (r) ──
    frozen, trainable = 0, 0
    for block in spnn.blocks:
        if isinstance(block, ConvPINNBlock):
            for name in ['s', 't', 'mix']:
                for p in getattr(block, name).parameters():
                    p.requires_grad = False
                    frozen += p.numel()
            for p in block.r.parameters():
                trainable += p.numel()
    if is_main:
        print(f"Frozen encoder params: {frozen:,}, trainable decoder params: {trainable:,}")

    # ── LPIPS (frozen) ──
    import lpips
    lpips_fn = None
    if args.lambda_lpips > 0:
        lpips_fn = lpips.LPIPS(net="vgg").to(device)
        lpips_fn.eval()
        for p in lpips_fn.parameters():
            p.requires_grad = False
        if is_main:
            print("LPIPS loss enabled")

    # ── Wrap decoder for DDP: forward() = decode() so DDP syncs r grads ──
    decoder = DecoderWrapper(spnn)

    # ── Optimizer (only r params) ──
    trainable_params = [p for p in decoder.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-5)

    # ── Accelerate prepare ──
    decoder, optimizer, train_loader = accelerator.prepare(
        decoder, optimizer, train_loader)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_epochs * len(train_loader), eta_min=1e-6)
    mse_loss = nn.MSELoss()

    # ── Fixed test images for visualization ──
    test_iter = iter(test_loader)
    test_images, _ = next(test_iter)
    test_images = test_images[:8].to(device)
    del test_iter

    # ── WandB ──
    if is_main:
        wandb.init(project="spnn-cifar10", name=f"finetune_decoder/{run_name}",
                   config=vars(args))

    best_loss = float('inf')
    ckpt_path = os.path.join(args.output_dir, "spnn_cifar10_best.pt")

    for epoch in range(1, args.num_epochs + 1):
        decoder.train()
        epoch_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.num_epochs}",
                    disable=not is_main)

        for batch_idx, (images, _labels) in enumerate(pbar):
            images = images.to(device)

            # ── VAE encode → sample from posterior ──
            with torch.no_grad():
                posterior = vae.encode(images)
                mu = posterior.mean
                std = posterior.std
                eps = torch.randn_like(mu)
                z = mu + args.alpha * std * eps

                # VAE decode for distillation target
                vae_decoded = vae.decode(z)

            # ── SPNN decode (single DDP forward for all decode paths) ──
            unwrapped = accelerator.unwrap_model(decoder).spnn
            if args.lambda_roundtrip > 0:
                with torch.no_grad():
                    z_spnn = unwrapped.encode(images)  # encoder is frozen
                # Concatenate both latents for a single DDP forward call
                z_cat = torch.cat([z, z_spnn], dim=0)
                decoded_cat = decoder(z_cat)
                spnn_decoded = decoded_cat[:z.size(0)]
                spnn_recon = decoded_cat[z.size(0):]
            else:
                spnn_decoded = decoder(z)
                spnn_recon = None

            # ── Distillation loss: match VAE decoder ──
            distill_loss = mse_loss(spnn_decoded, vae_decoded)

            # ── Reconstruction loss: match original image ──
            recon_loss = mse_loss(spnn_decoded, images)

            # ── LPIPS perceptual loss vs original ──
            lpips_loss = torch.tensor(0.0, device=device)
            if lpips_fn is not None:
                up_spnn = F.interpolate(spnn_decoded, size=64, mode='bilinear',
                                        align_corners=False)
                up_orig = F.interpolate(images, size=64, mode='bilinear',
                                        align_corners=False)
                lpips_loss = lpips_fn(up_spnn, up_orig).mean()

            # ── Roundtrip loss: decode(encode(x)) ≈ x ──
            roundtrip_loss = torch.tensor(0.0, device=device)
            if spnn_recon is not None:
                roundtrip_loss = mse_loss(spnn_recon, images)

            loss = (args.lambda_distill * distill_loss
                    + args.lambda_recon * recon_loss
                    + args.lambda_lpips * lpips_loss
                    + args.lambda_roundtrip * roundtrip_loss)

            optimizer.zero_grad()
            accelerator.backward(loss)
            if args.max_grad_norm > 0:
                grad_norm = accelerator.clip_grad_norm_(
                    decoder.parameters(), max_norm=args.max_grad_norm)
            else:
                grad_norm = accelerator.clip_grad_norm_(
                    decoder.parameters(), max_norm=float('inf'))
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()

            if is_main:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/distill_loss": distill_loss.item(),
                    "train/recon_loss": recon_loss.item(),
                    "train/lpips_loss": lpips_loss.item(),
                    "train/roundtrip_loss": roundtrip_loss.item(),
                    "train/lr": scheduler.get_last_lr()[0],
                    "train/grad_norm": grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm,
                })
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                })

        avg_loss = epoch_loss / len(train_loader)
        if is_main:
            wandb.log({"train/epoch_avg_loss": avg_loss, "epoch": epoch})
            print(f"  Epoch {epoch} — avg loss: {avg_loss:.6f}")

        # ── Visualization ──
        if epoch % args.save_every == 0 and is_main:
            decoder.eval()
            unwrapped_spnn = accelerator.unwrap_model(decoder).spnn
            with torch.no_grad():
                post = vae.encode(test_images)
                z_test = post.mean + args.alpha * post.std * torch.randn_like(post.mean)
                vae_dec = vae.decode(z_test)
                spnn_dec = unwrapped_spnn.decode(z_test)
            path = os.path.join(sample_dir, f"epoch{epoch:03d}.png")
            save_grid(spnn_dec, vae_dec, test_images, path)

        # ── Save best ──
        if avg_loss < best_loss:
            best_loss = avg_loss
            if is_main:
                unwrapped_spnn = accelerator.unwrap_model(decoder).spnn
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": unwrapped_spnn.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_loss,
                    "stages": CIFAR10_STAGES,
                }, ckpt_path)
                print(f"  New best loss: {avg_loss:.6f} — saved: {ckpt_path}")

    if is_main:
        print(f"\nFine-tuning complete. Best loss: {best_loss:.6f}")
        print(f"Best model: {ckpt_path}")
        wandb.finish()


def parse_args():
    p = argparse.ArgumentParser(
        description="Fine-tune SPNN decoder on sampled VAE latents")
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Pretrained SPNN checkpoint to fine-tune")
    p.add_argument("--num_epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--alpha", type=float, default=1.0,
                   help="Sampling spread: z = μ + α·σ·ε (1.0=standard, >1=wider)")
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--mix_type", type=str, default="cayley")
    p.add_argument("--scale_bound", type=float, default=2.0)
    p.add_argument("--lambda_distill", type=float, default=1.0,
                   help="Weight for MSE(SPNN.decode(z), VAE.decode(z))")
    p.add_argument("--lambda_recon", type=float, default=1.0,
                   help="Weight for MSE(SPNN.decode(z), x)")
    p.add_argument("--lambda_lpips", type=float, default=0.1,
                   help="Weight for LPIPS(SPNN.decode(z), x)")
    p.add_argument("--lambda_roundtrip", type=float, default=0.5,
                   help="Weight for MSE(SPNN.decode(SPNN.encode(x)), x)")
    p.add_argument("--max_grad_norm", type=float, default=5.0)
    p.add_argument("--save_every", type=int, default=5)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--output_dir", type=str,
                   default="cifar10_experiment/checkpoints_finetune_decoder")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
