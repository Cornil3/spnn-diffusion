import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from diffusers import AutoencoderKL
import wandb
import lpips
from models import SPNNAutoencoder
from dataset import CelebAHQDataset
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
    latent = posterior.mode()
    scaled_latent = latent * vae.config.scaling_factor
    decoded = vae.decode(latent).sample
    return scaled_latent, decoded

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

    # ── Fixed test batch for Penrose checks ──
    test_dataset = CelebAHQDataset(
        img_size=args.img_size, split="test", n_test=args.n_test,
    )
    test_loader = DataLoader(test_dataset, batch_size=args.penrose_batch_size, shuffle=False)
    penrose_images = next(iter(test_loader)).to(DEVICE)
    penrose_latent, _ = get_vae_pairs(vae, penrose_images)
    del test_dataset, test_loader
    print(f"Penrose check: using {penrose_images.size(0)} fixed test images")

    # ── Optimizer: trains ALL of s, t, r, mix through the decoder path ──
    optimizer = torch.optim.AdamW(spnn.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_epochs * len(loader), eta_min=1e-6
    )
    mse_loss = nn.MSELoss()

    for epoch in range(1, args.num_epochs + 1):
        spnn.train()
        epoch_loss = 0.0

        pbar = tqdm(loader, desc=f"Epoch {epoch}/{args.num_epochs}")
        for batch_idx, images in enumerate(pbar):
            images = images.to(DEVICE)

            # ── Get VAE targets: latent -> decoded image ──
            vae_latent, vae_decoded = get_vae_pairs(vae, images)

            # ── Train SPNN decoder: feed VAE latent, match VAE output ──
            # This is the pinv path: it uses r to estimate side-channels,
            # then s and t to invert the affine coupling, then mix.inverse.
            # All of s, t, r, mix get gradients here.
            spnn_decoded = spnn.decode(vae_latent)

            decoder_loss = mse_loss(spnn_decoded, vae_decoded)

            # ── LPIPS perceptual loss ──
            lpips_loss = torch.tensor(0.0, device=DEVICE)
            if lpips_fn is not None:
                lpips_loss = lpips_fn(spnn_decoded, vae_decoded).mean()

            loss = decoder_loss + args.lambda_lpips * lpips_loss

            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(spnn.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()

            wandb.log({
                "train/loss": loss.item(),
                "train/decoder_loss": decoder_loss.item(),
                "train/lpips_loss": lpips_loss.item(),
                "train/lr": scheduler.get_last_lr()[0],
                "train/grad_norm": grad_norm.item(),
            })

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}",
            })

            # if batch_idx % 200 == 0:
            #     save_comparison(spnn_decoded, vae_decoded, images, epoch, batch_idx, train_sample_dir)

        avg_loss = epoch_loss / len(loader)
        wandb.log({"train/epoch_avg_loss": avg_loss, "epoch": epoch})
        print(f"  Epoch {epoch} — avg decoder loss: {avg_loss:.6f}")

        # ── Penrose + roundtrip checks (diagnostic only) ──
        if epoch % args.save_every == 0:
            p_metrics = penrose_check(spnn, penrose_images, penrose_latent, DEVICE)
            print_penrose_metrics(p_metrics)
            wandb.log({**p_metrics, "epoch": epoch})
            spnn.train()

            ckpt_path = os.path.join(args.output_dir, f"spnn_vae_epoch{epoch:03d}.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": spnn.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
            }, ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}")

    # ── Final save ──
    final_path = os.path.join(args.output_dir, "spnn_vae_final.pt")
    torch.save(spnn.state_dict(), final_path)
    print(f"\nTraining complete. Final model: {final_path}")
    print(f"The encoder (spnn.encode / forward) now works automatically —")
    print(f"it uses the same s, t, mix that were trained through the decoder.")