"""
Train SPNN on CIFAR-10 using Simple LDM VAE as teacher.

SPNN architecture for CIFAR-10:
    3x32x32 -> PixelUnshuffle(2) -> 12x16x16 -> ConvPINN(12->3) -> 3x16x16

The teacher VAE (Simple LDM) maps 32x32 images to 3x16x16 latents.
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
from accelerate import Accelerator

# ── SPNN building blocks from parent project ──
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from models import ConvPINNBlock, PixelUnshuffleBlock
from diagnostics import penrose_check, print_penrose_metrics

# ── Simple LDM imports ──
SLDM_ROOT = os.path.join(os.path.dirname(__file__), '..',
    'simple-latent-diffusion-model-master', 'simple-latent-diffusion-model')
sys.path.insert(0, SLDM_ROOT)
from auto_encoder.models.variational_auto_encoder import VariationalAutoEncoder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONFIG_PATH = os.path.join(SLDM_ROOT, 'configs', 'cifar10_config.yaml')


# ═══════════════════════════════════════════════════════════
# Latent Discriminator for adversarial distribution matching
# ═══════════════════════════════════════════════════════════

class LatentDiscriminator(nn.Module):
    """
    Lightweight discriminator on latent space (e.g. 3x16x16).
    Uses spectral normalization for stable GAN training.
    Classifies latents as VAE-encoded (real) vs SPNN-encoded (fake).
    """
    def __init__(self, in_ch=3, spatial=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(in_ch, 64, 3, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),

            nn.utils.spectral_norm(nn.Conv2d(64, 128, 3, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),

            nn.utils.spectral_norm(nn.Conv2d(128, 256, 3, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),

            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.utils.spectral_norm(nn.Linear(256, 1)),
        )

    def forward(self, z):
        return self.net(z)


# ═══════════════════════════════════════════════════════════
# Configurable SPNN Autoencoder
# ═══════════════════════════════════════════════════════════

CIFAR10_STAGES = [('unshuffle', 2), ('pinn', 12, 3, 16)]
# 3x32x32 -> 12x16x16 -> 3x16x16

SD_STAGES = [('unshuffle', 4), ('pinn', 48, 16, 64),
             ('unshuffle', 2), ('pinn', 64, 4, 32)]
# 3x256x256 -> 48x64x64 -> 16x64x64 -> 64x32x32 -> 4x32x32


class SPNNAutoencoderConfigurable(nn.Module):
    """
    SPNN autoencoder with configurable stages.

    Each stage is either:
        ('unshuffle', r)                  -> PixelUnshuffleBlock(r)
        ('pinn', in_ch, out_ch, feat_size) -> ConvPINNBlock(in_ch, out_ch, ...)
    """

    def __init__(self, stages, mix_type="cayley", hidden=64,
                 scale_bound=2.0):
        super().__init__()
        blocks = []
        for stage in stages:
            if stage[0] == 'unshuffle':
                blocks.append(PixelUnshuffleBlock(stage[1]))
            elif stage[0] == 'pinn':
                _, in_ch, out_ch, feat_size = stage
                blocks.append(ConvPINNBlock(
                    in_ch, out_ch, hidden=hidden,
                    scale_bound=scale_bound, mix_type=mix_type,
                    feat_size=feat_size,
                ))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        """Forward pass = encode. Required for DDP wrapping."""
        return self.encode(x)

    def encode(self, x):
        for b in self.blocks:
            x = b(x)
        return x

    def decode(self, y):
        for b in reversed(self.blocks):
            y = b.pinv(y)
        return y


# ═══════════════════════════════════════════════════════════
# Simple LDM VAE loading
# ═══════════════════════════════════════════════════════════

def _load_checkpoint(model, path):
    """Load checkpoint robustly — handles EMA, model_state_dict, or raw state dict."""
    from helper.ema import EMA
    ckpt = torch.load(path, map_location=DEVICE, weights_only=True)
    # Try EMA keys (different checkpoints use different names)
    for ema_key in ('ema_state_dict', 'ema_model_state_dict'):
        if isinstance(ckpt, dict) and ema_key in ckpt:
            ema = EMA(model)
            ema.load_state_dict(ckpt[ema_key])
            return ema.ema_model
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
        return model
    else:
        # Raw state dict
        model.load_state_dict(ckpt)
        return model


def load_simple_vae():
    """Load frozen Simple LDM VAE from pretrained checkpoint."""
    return load_simple_vae_to(DEVICE)


def load_simple_vae_to(device):
    """Load frozen Simple LDM VAE from pretrained checkpoint to given device."""
    vae = VariationalAutoEncoder(CONFIG_PATH)
    model_path = os.path.join(SLDM_ROOT, 'models', 'cifar_vae.pth')
    ckpt = torch.load(model_path, map_location=device, weights_only=True)
    from helper.ema import EMA
    for ema_key in ('ema_state_dict', 'ema_model_state_dict'):
        if isinstance(ckpt, dict) and ema_key in ckpt:
            ema = EMA(vae)
            ema.load_state_dict(ckpt[ema_key])
            vae = ema.ema_model
            break
    else:
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            vae.load_state_dict(ckpt['model_state_dict'])
        else:
            vae.load_state_dict(ckpt)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False
    return vae.to(device)


@torch.no_grad()
def get_vae_pairs(vae, images):
    """
    Get (latent, decoded_image) from Simple LDM VAE.
    No scaling_factor — raw latents.
    """
    posterior = vae.encode(images)
    latent = posterior.mode()  # 3x16x16
    decoded = vae.decode(latent)
    return latent, decoded


# ═══════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════

def get_cifar10_loaders(batch_size, num_workers=4):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),  # -> [-1, 1]
    ])
    train_ds = datasets.CIFAR10(root='./data', train=True,
                                download=True, transform=transform)
    test_ds = datasets.CIFAR10(root='./data', train=False,
                               download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True,
                              drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader


# ═══════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════

def save_comparison(spnn_decoded, vae_decoded, original, epoch, batch_idx, sample_dir):
    n = min(4, original.size(0))
    orig = (original[:n].cpu() + 1) / 2
    vae_r = (vae_decoded[:n].cpu() + 1) / 2
    spnn_r = (spnn_decoded[:n].detach().cpu() + 1) / 2
    grid = torch.cat([orig, vae_r, spnn_r], dim=0)
    path = os.path.join(sample_dir, f"epoch{epoch:03d}_batch{batch_idx:04d}.png")
    save_image(grid, path, nrow=n, padding=2)


def train(args):
    # ── Accelerator (handles DDP, mixed precision, device placement) ──
    accelerator = Accelerator(mixed_precision='fp16')
    device = accelerator.device
    is_main = accelerator.is_main_process

    if is_main:
        print(f"Device: {device}  |  Num processes: {accelerator.num_processes}")

    # Build run name from loss lambdas
    run_name = (f"dec{args.lambda_decoder}_lp{args.lambda_lpips}_cy{args.lambda_cycle}"
                f"_rt{args.lambda_roundtrip}_al{args.lambda_align}"
                f"_mm{args.lambda_moment}_adv{args.lambda_adv}_h{args.hidden}")
    args.output_dir = os.path.join(args.output_dir, run_name)

    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)
        sample_dir = os.path.join(args.output_dir, "samples")
        os.makedirs(sample_dir, exist_ok=True)
        print(f"Output dir: {args.output_dir}")

    # ── Dataset ──
    train_loader, test_loader = get_cifar10_loaders(
        args.batch_size, args.num_workers)
    if is_main:
        print(f"CIFAR-10: {len(train_loader.dataset)} train, "
              f"{len(test_loader.dataset)} test, {len(train_loader)} batches/epoch")

    # ── Frozen models (not wrapped by accelerate — just move to device) ──
    vae = load_simple_vae_to(device)

    spnn = SPNNAutoencoderConfigurable(
        stages=CIFAR10_STAGES,
        mix_type=args.mix_type,
        hidden=args.hidden,
        scale_bound=args.scale_bound,
    ).to(device)

    if args.resume:
        if is_main:
            print(f"Resuming from {args.resume}...")
        state = torch.load(args.resume, map_location=device, weights_only=True)
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        spnn.load_state_dict(state)

    total_params = sum(p.numel() for p in spnn.parameters())
    if is_main:
        print(f"SPNN total params: {total_params:,}")

    # ── LPIPS (frozen, not wrapped) ──
    import lpips
    lpips_fn = None
    if args.lambda_lpips > 0:
        lpips_fn = lpips.LPIPS(net="vgg").to(device)
        lpips_fn.eval()
        for p in lpips_fn.parameters():
            p.requires_grad = False
        if is_main:
            print("LPIPS loss enabled (upsample to 64x64)")

    # ── Fixed test batch for Penrose checks ──
    test_iter = iter(test_loader)
    penrose_images, _ = next(test_iter)
    penrose_images = penrose_images[:args.penrose_batch_size].to(device)
    penrose_latent, _ = get_vae_pairs(vae, penrose_images)
    del test_iter
    if is_main:
        print(f"Penrose check: using {penrose_images.size(0)} fixed test images")

    # ── Adversarial latent discriminator (not wrapped — only SPNN gets DDP) ──
    use_adv = args.lambda_adv > 0
    disc = None
    disc_optimizer = None
    if use_adv:
        in_ch, spatial = penrose_latent.shape[1], penrose_latent.shape[2]
        disc = LatentDiscriminator(in_ch=in_ch, spatial=spatial).to(device)
        disc_optimizer = torch.optim.Adam(disc.parameters(), lr=args.lr_disc,
                                          betas=(0.0, 0.9))
        if is_main:
            d_params = sum(p.numel() for p in disc.parameters())
            print(f"Adversarial loss enabled: discriminator {d_params:,} params")

    # ── Freeze params if requested ──
    if args.freeze:
        from models import ConvPINNBlock
        frozen, trainable = 0, 0
        for block in spnn.blocks:
            if isinstance(block, ConvPINNBlock):
                if args.freeze == "r":
                    # Freeze r, train s/t/mix
                    for p in block.r.parameters():
                        p.requires_grad = False
                        frozen += p.numel()
                    for name in ['s', 't', 'mix']:
                        for p in getattr(block, name).parameters():
                            trainable += p.numel()
                elif args.freeze == "encoder":
                    # Freeze s/t/mix, train r only
                    for name in ['s', 't', 'mix']:
                        for p in getattr(block, name).parameters():
                            p.requires_grad = False
                            frozen += p.numel()
                    for p in block.r.parameters():
                        trainable += p.numel()
        if is_main:
            print(f"Freeze mode '{args.freeze}': frozen={frozen:,}, trainable={trainable:,}")

    # ── Optimizer (only trainable params) ──
    trainable_params = [p for p in spnn.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-5)

    # ── Accelerate prepare (wraps model with DDP, splits dataloader) ──
    spnn, optimizer, train_loader = accelerator.prepare(
        spnn, optimizer, train_loader)

    # ── Scheduler (created AFTER prepare so len(train_loader) is per-GPU) ──
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_epochs * len(train_loader), eta_min=1e-6)
    mse_loss = nn.MSELoss()

    # ── WandB (rank 0 only) ──
    if is_main:
        wandb.init(project="spnn-cifar10", name=f"train_cifar10/{run_name}",
                   config=vars(args))

    best_loss = float('inf')
    ckpt_path = os.path.join(args.output_dir, "spnn_cifar10_best.pt")

    for epoch in range(1, args.num_epochs + 1):
        spnn.train()
        epoch_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.num_epochs}",
                    disable=not is_main)
        unwrapped = accelerator.unwrap_model(spnn)
        for batch_idx, (images, _labels) in enumerate(pbar):
            images = images.to(device)

            # ── VAE targets ──
            vae_latent, vae_decoded = get_vae_pairs(vae, images)

            # ── Encode through DDP (arms gradient sync reducer) ──
            z_spnn = spnn(images)

            # ── Decoder loss: feed VAE latent, match VAE output ──
            spnn_decoded = unwrapped.decode(vae_latent)
            decoder_loss = mse_loss(spnn_decoded, vae_decoded)

            # ── LPIPS (upsample to 64x64) ──
            lpips_loss = torch.tensor(0.0, device=device)
            if lpips_fn is not None:
                up_spnn = F.interpolate(spnn_decoded, size=64, mode='bilinear',
                                        align_corners=False)
                up_vae = F.interpolate(vae_decoded, size=64, mode='bilinear',
                                       align_corners=False)
                lpips_loss = lpips_fn(up_spnn, up_vae).mean()

            # ── Cycle loss: encode(decode(z)) ≈ z ──
            cycle_loss = torch.tensor(0.0, device=device)
            if args.lambda_cycle > 0:
                re_encoded = unwrapped.encode(spnn_decoded)
                cycle_loss = mse_loss(re_encoded, vae_latent)

            # ── Roundtrip loss: decode(encode(x)) ≈ x ──
            roundtrip_loss = torch.tensor(0.0, device=device)
            if args.lambda_roundtrip > 0:
                spnn_recon = unwrapped.decode(z_spnn)
                roundtrip_loss = mse_loss(spnn_recon, images)

            # ── Encoder latents for align + moment + adversarial ──
            align_loss = torch.tensor(0.0, device=device)
            moment_loss = torch.tensor(0.0, device=device)
            d_loss = torch.tensor(0.0, device=device)
            g_loss = torch.tensor(0.0, device=device)

            # Latent alignment: SPNN.encode(x) ≈ VAE.encode(x)
            if args.lambda_align > 0:
                align_loss = mse_loss(z_spnn, vae_latent)

            # Moment matching: match per-channel mean and variance
            if args.lambda_moment > 0:
                moment_loss = (
                    F.mse_loss(z_spnn.mean([0, 2, 3]), vae_latent.mean([0, 2, 3]))
                    + F.mse_loss(z_spnn.var([0, 2, 3]), vae_latent.var([0, 2, 3]))
                )

            # Adversarial: train D then compute G loss
            if use_adv:
                # (a) Discriminator step
                disc_optimizer.zero_grad()
                d_real = disc(vae_latent.detach())
                d_fake = disc(z_spnn.detach())
                d_loss = (F.relu(1.0 - d_real).mean()
                          + F.relu(1.0 + d_fake).mean())
                d_loss.backward()
                disc_optimizer.step()

                # (b) Generator loss for encoder
                g_loss = -disc(z_spnn).mean()

            loss = (args.lambda_decoder * decoder_loss
                    + args.lambda_lpips * lpips_loss
                    + args.lambda_cycle * cycle_loss
                    + args.lambda_roundtrip * roundtrip_loss
                    + args.lambda_align * align_loss
                    + args.lambda_moment * moment_loss
                    + args.lambda_adv * g_loss)

            optimizer.zero_grad()
            accelerator.backward(loss)
            if args.max_grad_norm > 0:
                grad_norm = accelerator.clip_grad_norm_(
                    spnn.parameters(), max_norm=args.max_grad_norm)
            else:
                grad_norm = accelerator.clip_grad_norm_(
                    spnn.parameters(), max_norm=float('inf'))
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()

            if is_main:
                log_dict = {
                    "train/loss": loss.item(),
                    "train/decoder_loss": decoder_loss.item(),
                    "train/lpips_loss": lpips_loss.item(),
                    "train/cycle_loss": cycle_loss.item(),
                    "train/roundtrip_loss": roundtrip_loss.item(),
                    "train/align_loss": align_loss.item(),
                    "train/moment_loss": moment_loss.item(),
                    "train/lr": scheduler.get_last_lr()[0],
                    "train/grad_norm": grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm,
                }
                if use_adv:
                    log_dict["train/d_loss"] = d_loss.item()
                    log_dict["train/g_loss"] = g_loss.item()
                wandb.log(log_dict)

                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                })

        avg_loss = epoch_loss / len(train_loader)
        if is_main:
            wandb.log({"train/epoch_avg_loss": avg_loss, "epoch": epoch})
            print(f"  Epoch {epoch} — avg loss: {avg_loss:.6f}")

        # ── Penrose diagnostics (rank 0 only) ──
        if epoch % args.save_every == 0 and is_main:
            unwrapped_spnn = accelerator.unwrap_model(spnn)
            p_metrics = penrose_check(unwrapped_spnn, penrose_images, penrose_latent, device)
            print_penrose_metrics(p_metrics)
            wandb.log({**p_metrics, "epoch": epoch})
            spnn.train()

            # Save comparison images
            with torch.no_grad():
                sample_decoded = unwrapped_spnn.decode(penrose_latent[:4])
                _, vae_dec = get_vae_pairs(vae, penrose_images[:4])
            save_comparison(sample_decoded, vae_dec, penrose_images[:4],
                            epoch, 0, sample_dir)

        # ── Save best model (rank 0 only, unwrapped state dict) ──
        if avg_loss < best_loss:
            best_loss = avg_loss
            if is_main:
                unwrapped_spnn = accelerator.unwrap_model(spnn)
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": unwrapped_spnn.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_loss,
                    "stages": CIFAR10_STAGES,
                }, ckpt_path)
                print(f"  New best loss: {avg_loss:.6f} — saved: {ckpt_path}")

    if is_main:
        print(f"\nTraining complete. Best loss: {best_loss:.6f}")
        print(f"Best model: {ckpt_path}")
        wandb.finish()


def parse_args():
    p = argparse.ArgumentParser(description="Train SPNN on CIFAR-10")
    p.add_argument("--num_epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--mix_type", type=str, default="cayley",
                   choices=["cayley", "householder"])
    p.add_argument("--scale_bound", type=float, default=2.0)
    p.add_argument("--lambda_decoder", type=float, default=1.0,
                   help="Decoder MSE loss weight")
    p.add_argument("--lambda_lpips", type=float, default=0.1)
    p.add_argument("--lambda_cycle", type=float, default=1.0)
    p.add_argument("--lambda_roundtrip", type=float, default=1.0)
    p.add_argument("--lambda_align", type=float, default=0.1)
    p.add_argument("--lambda_moment", type=float, default=0.0,
                   help="Moment matching weight (match per-channel mean+var)")
    p.add_argument("--lambda_adv", type=float, default=0.0,
                   help="Adversarial latent matching weight (0=disabled)")
    p.add_argument("--lr_disc", type=float, default=1e-4,
                   help="Discriminator learning rate")
    p.add_argument("--max_grad_norm", type=float, default=5.0)
    p.add_argument("--save_every", type=int, default=5)
    p.add_argument("--penrose_batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--freeze", type=str, default=None,
                   choices=["r", "encoder"],
                   help="Freeze params: 'r' = freeze r networks (train s,t,mix), "
                        "'encoder' = freeze s,t,mix (train r only)")
    p.add_argument("--resume", type=str, default=None,
                   help="Path to pretrained SPNN checkpoint to fine-tune from")
    p.add_argument("--output_dir", type=str,
                   default="cifar10_experiment/checkpoints")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
