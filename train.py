import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from diffusers import AutoencoderKL
import wandb
import lpips
from accelerate import Accelerator
from models import SPNNAutoencoder
from dataset import CelebAHQDataset, LAIONAestheticDataset
from diagnostics import penrose_check, print_penrose_metrics

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_sd_vae(device, verbose=True):
    """Load frozen SD-VAE to given device."""
    if verbose:
        print("Loading VAE from timbrooks/instruct-pix2pix...")
    vae = AutoencoderKL.from_pretrained(
        "timbrooks/instruct-pix2pix", subfolder="vae"
    )
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False
    return vae.to(device)


@torch.no_grad()
def get_vae_pairs(vae, images):
    """
    Get (latent, decoded_image) pairs from the frozen SD-VAE.
    These are the training targets for our SPNN decoder.
    """
    posterior = vae.encode(images).latent_dist
    latent = posterior.sample()
    decoded = vae.decode(latent).sample
    return latent, decoded

def save_cycle_comparison(spnn, images, epoch, sample_dir, num_cycles=5):
    """Save grid: row 0 = original, rows 1..num_cycles = after each encode→decode cycle."""
    from torchvision.utils import save_image
    spnn.eval()
    n = min(4, images.size(0))
    x = images[:n]
    rows = [(x.cpu() + 1) / 2]
    for _ in range(num_cycles):
        x = spnn.decode(spnn.encode(x))
        rows.append((x.detach().cpu() + 1) / 2)
    grid = torch.cat(rows, dim=0)
    path = os.path.join(sample_dir, f"epoch{epoch:03d}_cycles.png")
    save_image(grid, path, nrow=n, padding=2)

def train(args):
    # ── Accelerator (handles DDP, mixed precision, device placement) ──
    accelerator = Accelerator(mixed_precision='bf16')
    device = accelerator.device
    is_main = accelerator.is_main_process

    if is_main:
        print(f"Device: {device}  |  Num processes: {accelerator.num_processes}")

    os.makedirs(args.output_dir, exist_ok=True)
    train_sample_dir = os.path.join(args.sample_dir, "train")
    if is_main:
        os.makedirs(train_sample_dir, exist_ok=True)

    if args.dataset == "laion":
        dataset = LAIONAestheticDataset(
            data_dir=args.laion_dir, img_size=args.img_size,
            split="train", n_test=args.n_test, max_images=args.max_images,
        )
    else:
        dataset = CelebAHQDataset(
            img_size=args.img_size, max_images=args.max_images,
            split="train", n_test=args.n_test,
        )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True
    )
    if is_main:
        print(f"Dataset: {len(dataset)} images, {len(loader)} batches/epoch")

    # ── Frozen models (not wrapped by accelerate — just move to device) ──
    vae = load_sd_vae(device, verbose=is_main)
    spnn = SPNNAutoencoder(mix_type=args.mix_type, hidden=args.hidden,
                           r_hidden=args.hidden,
                           scale_bound=args.scale_bound).to(device)

    total_params = sum(p.numel() for p in spnn.parameters())
    if is_main:
        print(f"SPNN total params: {total_params:,}")

    # ── LPIPS perceptual loss (frozen, not wrapped) ──
    lpips_fn = None
    if args.lambda_lpips > 0:
        lpips_fn = lpips.LPIPS(net="vgg").to(device)
        lpips_fn.eval()
        for p in lpips_fn.parameters():
            p.requires_grad = False
        if is_main:
            print("LPIPS loss enabled (VGG backbone)")

    # ── Test dataset for Penrose checks (rank 0 only) ──
    penrose_test_dataset = None
    if is_main:
        if args.dataset == "laion":
            penrose_test_dataset = LAIONAestheticDataset(
                data_dir=args.laion_dir, img_size=args.img_size,
                split="test", n_test=args.n_test,
            )
        else:
            penrose_test_dataset = CelebAHQDataset(
                img_size=args.img_size, split="test", n_test=args.n_test,
            )
        print(f"Penrose check: {len(penrose_test_dataset)} test images, "
              f"sampling {args.penrose_batch_size} random each time")

    # ── Optimizer: trains ALL of s, t, r, mix ──
    optimizer = torch.optim.AdamW(spnn.parameters(), lr=args.lr, weight_decay=1e-5)

    # ── Accelerate prepare (wraps model with DDP, splits dataloader) ──
    spnn, optimizer, loader = accelerator.prepare(spnn, optimizer, loader)

    # ── Scheduler (created AFTER prepare so len(loader) is per-GPU) ──
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_epochs * len(loader), eta_min=1e-6
    )
    mse_loss = nn.MSELoss()
    best_loss = float('inf')
    best_ckpt_path = os.path.join(args.output_dir, "spnn_vae_best.pt")

    # ── WandB (rank 0 only) ──
    if is_main:
        wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                   name=args.wandb_run_name, config=vars(args))

    for epoch in range(1, args.num_epochs + 1):
        spnn.train()
        epoch_loss = 0.0

        pbar = tqdm(loader, desc=f"Epoch {epoch}/{args.num_epochs}",
                    disable=not is_main)
        unwrapped = accelerator.unwrap_model(spnn)

        for batch_idx, images in enumerate(pbar):
            images = images.to(device)

            # ── Get VAE targets: latent -> decoded image ──
            vae_latent, vae_decoded = get_vae_pairs(vae, images)

            # ── Encode through DDP (arms gradient sync reducer) ──
            z_spnn = spnn(images)

            # ── Decoder distillation loss: feed VAE latent, match VAE output ──
            spnn_decoded = unwrapped.decode(vae_latent)
            decoder_distill_loss = mse_loss(spnn_decoded, vae_decoded)

            # ── Decoder GT loss: feed VAE latent, match original image ──
            decoder_gt_loss = torch.tensor(0.0, device=device)
            if args.lambda_decoder_gt > 0:
                decoder_gt_loss = mse_loss(spnn_decoded, images)

            # ── LPIPS perceptual loss ──
            lpips_loss = torch.tensor(0.0, device=device)
            if lpips_fn is not None:
                lpips_target = images if args.lambda_decoder_gt > 0 else vae_decoded
                lpips_loss = lpips_fn(spnn_decoded, lpips_target).mean()

            # ── Cycle loss (surjectivity): encode(decode(z)) ≈ z ──
            cycle_loss = torch.tensor(0.0, device=device)
            if args.lambda_cycle > 0:
                re_encoded = unwrapped.encode(spnn_decoded)
                cycle_loss = mse_loss(re_encoded, vae_latent)

            # ── Roundtrip loss (pseudo-inverse stability): decode(encode(x)) ≈ x ──
            roundtrip_loss = torch.tensor(0.0, device=device)
            if args.lambda_roundtrip > 0:
                spnn_recon = unwrapped.decode(z_spnn)
                roundtrip_loss = mse_loss(spnn_recon, images)

            # ── Latent alignment loss: SPNN.encode(x) ≈ VAE.encode(x) ──
            align_loss = torch.tensor(0.0, device=device)
            if args.lambda_align > 0:
                with torch.no_grad():
                    z_vae = vae.encode(images).latent_dist.mode()
                align_loss = mse_loss(z_spnn, z_vae)

            loss = (args.lambda_decoder_distill * decoder_distill_loss
                    + args.lambda_decoder_gt * decoder_gt_loss
                    + args.lambda_lpips * lpips_loss
                    + args.lambda_cycle * cycle_loss
                    + args.lambda_roundtrip * roundtrip_loss
                    + args.lambda_align * align_loss)

            optimizer.zero_grad()
            accelerator.backward(loss)
            if args.max_grad_norm > 0:
                grad_norm = accelerator.clip_grad_norm_(
                    spnn.parameters(), max_norm=args.max_grad_norm)
            else:
                grad_norm = accelerator.clip_grad_norm_(
                    spnn.parameters(), max_norm=float('inf'))

            # Skip optimizer step if loss or gradients are NaN
            if torch.isfinite(loss) and torch.isfinite(torch.tensor(grad_norm)):
                optimizer.step()
                scheduler.step()
            elif is_main:
                print(f"  [!] Skipping step {batch_idx} — NaN detected (loss={loss.item()}, grad_norm={grad_norm})")

            epoch_loss += loss.item()

            if is_main:
                log_dict = {
                    "train/loss": loss.item(),
                    "train/decoder_distill_loss": decoder_distill_loss.item(),
                    "train/decoder_gt_loss": decoder_gt_loss.item(),
                    "train/lpips_loss": lpips_loss.item(),
                    "train/cycle_loss": cycle_loss.item(),
                    "train/roundtrip_loss": roundtrip_loss.item(),
                    "train/align_loss": align_loss.item(),
                    "train/lr": scheduler.get_last_lr()[0],
                    "train/grad_norm": grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm,
                }
                wandb.log(log_dict)

                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                })

        avg_loss = epoch_loss / len(loader)
        if is_main:
            wandb.log({"train/epoch_avg_loss": avg_loss, "epoch": epoch})
            print(f"  Epoch {epoch} — avg decoder loss: {avg_loss:.6f}")

        # ── Save best model (rank 0 only, unwrapped state dict) ──
        if avg_loss < best_loss:
            best_loss = avg_loss
            if is_main:
                unwrapped_spnn = accelerator.unwrap_model(spnn)
                ckpt_dict = {
                    "epoch": epoch,
                    "model_state_dict": unwrapped_spnn.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_loss,
                }
                torch.save(ckpt_dict, best_ckpt_path)
                print(f"  New best loss: {avg_loss:.6f} — saved: {best_ckpt_path}")

        # ── Penrose + roundtrip checks (rank 0 only) ──
        if epoch % args.save_every == 0 and is_main:
            unwrapped_spnn = accelerator.unwrap_model(spnn)
            # Sample a fresh random batch each time
            penrose_loader = DataLoader(
                penrose_test_dataset, batch_size=args.penrose_batch_size,
                shuffle=True, num_workers=0)
            penrose_images = next(iter(penrose_loader)).to(device)
            with torch.no_grad():
                # Encode in chunks to avoid VAE OOM on large batches
                chunk_size = 16
                latent_chunks = []
                for i in range(0, penrose_images.size(0), chunk_size):
                    chunk = penrose_images[i:i+chunk_size]
                    latent_chunks.append(vae.encode(chunk).latent_dist.mode())
                penrose_latent = torch.cat(latent_chunks, dim=0)
            del penrose_loader
            p_metrics = penrose_check(unwrapped_spnn, penrose_images, penrose_latent, device)
            print_penrose_metrics(p_metrics)
            wandb.log({**p_metrics, "epoch": epoch})

            # Save cycle consistency grid (5 encode→decode cycles)
            with torch.no_grad():
                save_cycle_comparison(unwrapped_spnn, penrose_images, epoch,
                                     train_sample_dir, num_cycles=5)

            spnn.train()

            ckpt_path = os.path.join(args.output_dir, f"spnn_vae_epoch{epoch:03d}.pt")
            ckpt_dict = {
                "epoch": epoch,
                "model_state_dict": unwrapped_spnn.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
            }
            torch.save(ckpt_dict, ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}")

    # ── Final save (rank 0 only) ──
    if is_main:
        unwrapped_spnn = accelerator.unwrap_model(spnn)
        final_path = os.path.join(args.output_dir, "spnn_vae_final.pt")
        torch.save(unwrapped_spnn.state_dict(), final_path)
        print(f"\nTraining complete. Final model: {final_path}")
        print(f"The encoder (spnn.encode / forward) now works automatically —")
        print(f"it uses the same s, t, mix that were trained through the decoder.")
        wandb.finish()
