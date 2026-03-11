import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from diffusers import AutoencoderKL
import wandb
import lpips
from models import SPNNAutoencoder, PatchDiscWithContext
from dataset import CelebAHQDataset, LAIONAestheticDataset
from diagnostics import penrose_check, print_penrose_metrics

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"




def load_sd_vae():
    print("Loading VAE from timbrooks/instruct-pix2pix...")
    vae = AutoencoderKL.from_pretrained(
        "timbrooks/instruct-pix2pix", subfolder="vae"
    )
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False
    return vae.to(DEVICE)


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

def save_comparison(spnn_decoded, vae_decoded, original, epoch, batch_idx, sample_dir):
    from torchvision.utils import save_image
    n = min(4, original.size(0))
    orig = (original[:n].cpu() + 1) / 2
    vae_r = (vae_decoded[:n].cpu() + 1) / 2
    spnn_r = (spnn_decoded[:n].detach().cpu() + 1) / 2
    grid = torch.cat([orig, vae_r, spnn_r], dim=0)
    path = os.path.join(sample_dir, f"epoch{epoch:03d}_batch{batch_idx:04d}.png")
    save_image(grid, path, nrow=n, padding=2)

def train(args):
    print(f"Device: {DEVICE}")

    os.makedirs(args.output_dir, exist_ok=True)
    train_sample_dir = os.path.join(args.sample_dir, "train")
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
    print(f"Dataset: {len(dataset)} images, {len(loader)} batches/epoch")

    # ── Models ──
    vae = load_sd_vae()
    spnn = SPNNAutoencoder(mix_type=args.mix_type, hidden=args.hidden, scale_bound=args.scale_bound).to(DEVICE)

    total_params = sum(p.numel() for p in spnn.parameters())
    print(f"SPNN total params: {total_params:,}")

    # ── LPIPS perceptual loss (frozen) ──
    lpips_fn = None
    if args.lambda_lpips > 0:
        lpips_fn = lpips.LPIPS(net="vgg").to(DEVICE)
        lpips_fn.eval()
        for p in lpips_fn.parameters():
            p.requires_grad = False
        print("LPIPS loss enabled (VGG backbone)")

    # ── Seraena-style conditional discriminator ──
    discriminator = None
    d_optimizer = None
    d_scheduler = None
    replay_buffer = []
    max_buffer_len = 16384
    if args.lambda_gan > 0:
        discriminator = PatchDiscWithContext().to(DEVICE)
        d_params = sum(p.numel() for p in discriminator.parameters())
        print(f"Seraena PatchGAN discriminator enabled ({d_params:,} params)")
        d_optimizer = torch.optim.AdamW(
            discriminator.parameters(), lr=args.lr * 10, betas=(0.9, 0.99), weight_decay=1e-5
        )
        d_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            d_optimizer, T_max=args.num_epochs * len(loader), eta_min=1e-6
        )

    # ── Fixed test batch for Penrose checks ──
    if args.dataset == "laion":
        test_dataset = LAIONAestheticDataset(
            data_dir=args.laion_dir, img_size=args.img_size,
            split="test", n_test=args.n_test,
        )
    else:
        test_dataset = CelebAHQDataset(
            img_size=args.img_size, split="test", n_test=args.n_test,
        )
    test_loader = DataLoader(test_dataset, batch_size=args.penrose_batch_size, shuffle=False)
    penrose_images = next(iter(test_loader)).to(DEVICE)
    with torch.no_grad():
        penrose_latent = vae.encode(penrose_images).latent_dist.mode()
    del test_dataset, test_loader
    print(f"Penrose check: using {penrose_images.size(0)} fixed test images")

    # ── Optimizer: trains ALL of s, t, r, mix through the decoder path ──
    optimizer = torch.optim.AdamW(spnn.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_epochs * len(loader), eta_min=1e-6
    )
    mse_loss = nn.MSELoss()
    best_loss = float('inf')
    best_ckpt_path = os.path.join(args.output_dir, "spnn_vae_best.pt")

    for epoch in range(1, args.num_epochs + 1):
        spnn.train()
        epoch_loss = 0.0

        pbar = tqdm(loader, desc=f"Epoch {epoch}/{args.num_epochs}")
        for batch_idx, images in enumerate(pbar):
            images = images.to(DEVICE)

            # ── Get VAE targets: latent -> decoded image ──
            vae_latent, vae_decoded = get_vae_pairs(vae, images)

            # ── Decoder loss: feed VAE latent, match VAE output ──
            spnn_decoded = spnn.decode(vae_latent)
            decoder_loss = mse_loss(spnn_decoded, vae_decoded)

            # ── Seraena Phase A: Update discriminator with replay buffer ──
            d_loss = torch.tensor(0.0, device=DEVICE)
            g_loss = torch.tensor(0.0, device=DEVICE)
            if discriminator is not None:
                fake_detached = spnn_decoded.detach()
                ctx = vae_latent.detach()

                # Update replay buffer on CPU (reservoir sampling)
                for fi, ci in zip(fake_detached, ctx):
                    if len(replay_buffer) >= max_buffer_len:
                        i = random.randrange(0, len(replay_buffer))
                        replay_buffer[i][0].copy_(fi.cpu())
                        replay_buffer[i][1].copy_(ci.cpu())
                    else:
                        replay_buffer.append((fi.cpu().clone(), ci.cpu().clone()))

                # Build disc batch: half current, half from buffer
                n = len(fake_detached) // 2
                if len(replay_buffer) >= n:
                    buf_samples = [random.choice(replay_buffer) for _ in range(n)]
                    buf_fake = torch.stack([s[0] for s in buf_samples]).to(DEVICE)
                    buf_ctx = torch.stack([s[1] for s in buf_samples]).to(DEVICE)
                    disc_fake = torch.cat([fake_detached[:n], buf_fake], 0)
                    disc_ctx = torch.cat([ctx[:n], buf_ctx], 0)
                else:
                    disc_fake = fake_detached
                    disc_ctx = ctx

                # Expand real to match disc batch size
                disc_real = vae_decoded[:disc_fake.size(0)].detach()
                disc_real_ctx = ctx[:disc_fake.size(0)]

                # LSGAN disc step: random real/fake mixing per sample
                discriminator.train()
                fake_mask = (torch.rand(disc_fake.size(0), 1, 1, 1, device=DEVICE) < 0.5)
                in_ims = fake_mask.float() * disc_fake + (~fake_mask).float() * disc_real
                in_ctx = fake_mask.float() * disc_ctx + (~fake_mask).float() * disc_real_ctx
                scores = discriminator(in_ims, in_ctx)
                targets = fake_mask.float().mul(2).sub(1).expand_as(scores)
                d_loss = F.mse_loss(scores, targets)

                d_optimizer.zero_grad()
                d_loss.backward()
                d_optimizer.step()
                d_scheduler.step()

                # Seraena Phase B: Compute correction targets for generator
                discriminator.eval()
                correction = torch.zeros_like(spnn_decoded).requires_grad_(True)
                with torch.no_grad():
                    ref_feats = discriminator(vae_decoded, vae_latent)
                corr_feats = discriminator(spnn_decoded.detach() + correction, vae_latent)
                feat_loss = F.mse_loss(
                    ref_feats, corr_feats, reduction="none"
                ).mean((1, 2, 3), keepdim=True)
                feat_loss.sum().backward(inputs=[correction])
                corr_grad = correction.grad.detach().neg()
                corr_grad.div_(corr_grad.std() + 1e-5)
                correction_target = (spnn_decoded.detach() + 0.1 * corr_grad).detach()

                # Generator GAN loss: MSE toward correction target
                g_loss = mse_loss(spnn_decoded, correction_target)

            # ── LPIPS perceptual loss ──
            lpips_loss = torch.tensor(0.0, device=DEVICE)
            if lpips_fn is not None:
                lpips_loss = lpips_fn(spnn_decoded, vae_decoded).mean()

            # ── Cycle loss (surjectivity): encode(decode(z)) ≈ z ──
            cycle_loss = torch.tensor(0.0, device=DEVICE)
            if args.lambda_cycle > 0:
                re_encoded = spnn.encode(spnn_decoded)
                cycle_loss = mse_loss(re_encoded, vae_latent)

            # ── Roundtrip loss (pseudo-inverse stability): decode(encode(x)) ≈ x ──
            roundtrip_loss = torch.tensor(0.0, device=DEVICE)
            if args.lambda_roundtrip > 0:
                spnn_latent = spnn.encode(images)
                spnn_recon = spnn.decode(spnn_latent)
                roundtrip_loss = mse_loss(spnn_recon, images)

            # ── Latent alignment loss: SPNN.encode(x) ≈ VAE.encode(x) ──
            align_loss = torch.tensor(0.0, device=DEVICE)
            if args.lambda_align > 0:
                with torch.no_grad():
                    z_vae = vae.encode(images).latent_dist.mode()
                z_spnn = spnn.encode(images)
                align_loss = mse_loss(z_spnn, z_vae)

            loss = (decoder_loss
                    + args.lambda_lpips * lpips_loss
                    + args.lambda_cycle * cycle_loss
                    + args.lambda_roundtrip * roundtrip_loss
                    + args.lambda_gan * g_loss
                    + args.lambda_align * align_loss)

            optimizer.zero_grad()
            loss.backward()
            if args.max_grad_norm > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(spnn.parameters(), max_norm=args.max_grad_norm)
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(spnn.parameters(), max_norm=float('inf'))
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()

            log_dict = {
                "train/loss": loss.item(),
                "train/decoder_loss": decoder_loss.item(),
                "train/lpips_loss": lpips_loss.item(),
                "train/cycle_loss": cycle_loss.item(),
                "train/roundtrip_loss": roundtrip_loss.item(),
                "train/align_loss": align_loss.item(),
                "train/lr": scheduler.get_last_lr()[0],
                "train/grad_norm": grad_norm.item(),
            }
            if discriminator is not None:
                log_dict["train/d_loss"] = d_loss.item()
                log_dict["train/g_loss"] = g_loss.item()
            wandb.log(log_dict)

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}",
            })

            # if batch_idx % 200 == 0:
            #     save_comparison(spnn_decoded, vae_decoded, images, epoch, batch_idx, train_sample_dir)

        avg_loss = epoch_loss / len(loader)
        wandb.log({"train/epoch_avg_loss": avg_loss, "epoch": epoch})
        print(f"  Epoch {epoch} — avg decoder loss: {avg_loss:.6f}")

        # ── Save best model ──
        if avg_loss < best_loss:
            best_loss = avg_loss
            ckpt_dict = {
                "epoch": epoch,
                "model_state_dict": spnn.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
            }
            if discriminator is not None:
                ckpt_dict["discriminator_state_dict"] = discriminator.state_dict()
                ckpt_dict["d_optimizer_state_dict"] = d_optimizer.state_dict()
                ckpt_dict["d_scheduler_state_dict"] = d_scheduler.state_dict()
            torch.save(ckpt_dict, best_ckpt_path)
            print(f"  New best loss: {avg_loss:.6f} — saved: {best_ckpt_path}")

        # ── Penrose + roundtrip checks (diagnostic only) ──
        if epoch % args.save_every == 0:
            p_metrics = penrose_check(spnn, penrose_images, penrose_latent, DEVICE)
            print_penrose_metrics(p_metrics)
            wandb.log({**p_metrics, "epoch": epoch})
            spnn.train()

            ckpt_path = os.path.join(args.output_dir, f"spnn_vae_epoch{epoch:03d}.pt")
            ckpt_dict = {
                "epoch": epoch,
                "model_state_dict": spnn.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
            }
            if discriminator is not None:
                ckpt_dict["discriminator_state_dict"] = discriminator.state_dict()
                ckpt_dict["d_optimizer_state_dict"] = d_optimizer.state_dict()
                ckpt_dict["d_scheduler_state_dict"] = d_scheduler.state_dict()
            torch.save(ckpt_dict, ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}")

    # ── Final save ──
    final_path = os.path.join(args.output_dir, "spnn_vae_final.pt")
    torch.save(spnn.state_dict(), final_path)
    print(f"\nTraining complete. Final model: {final_path}")
    print(f"The encoder (spnn.encode / forward) now works automatically —")
    print(f"it uses the same s, t, mix that were trained through the decoder.")