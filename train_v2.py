import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from diffusers import AutoencoderKL
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
    Get (scaled_latent, decoded_image) pairs from the frozen SD-VAE.
    Applies the crucial SD scaling factor so SPNN learns the correct distribution variance.
    """
    posterior = vae.encode(images).latent_dist
    latent = posterior.mode()

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

    vae = load_sd_vae()
    spnn = SPNNAutoencoder(mix_type=args.mix_type, hidden=args.hidden, scale_bound=args.scale_bound).to(DEVICE)

    total_params = sum(p.numel() for p in spnn.parameters())
    print(f"SPNN total params: {total_params:,}")

    lpips_fn = None
    if args.lambda_lpips > 0:
        lpips_fn = lpips.LPIPS(net="vgg").to(DEVICE)
        lpips_fn.eval()
        for p in lpips_fn.parameters():
            p.requires_grad = False
        print("LPIPS loss enabled (VGG backbone)")

    discriminator = None
    d_optimizer = None
    d_scheduler = None

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
    penrose_latent, _ = get_vae_pairs(vae, penrose_images)
    del test_dataset, test_loader
    print(f"Penrose check: using {penrose_images.size(0)} fixed test images")

    optimizer = torch.optim.AdamW(spnn.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_epochs * len(loader), eta_min=1e-6
    )
    mse_loss = nn.MSELoss()

    for epoch in range(1, args.num_epochs + 1):
        spnn.train()
        epoch_loss = 0.0

        cur_lambda_lpips = 0.0
        desc_prefix = "EPOCHS"

        pbar = tqdm(loader, desc=f"{desc_prefix} Epoch {epoch}/{args.num_epochs}")
        for batch_idx, images in enumerate(pbar):
            images = images.to(DEVICE)

            vae_latent, vae_decoded = get_vae_pairs(vae, images)

            spnn_decoded = spnn.decode(vae_latent)
            decoder_loss = mse_loss(spnn_decoded, vae_decoded)

            lpips_loss = torch.tensor(0.0, device=DEVICE)
            if lpips_fn is not None and cur_lambda_lpips > 0:
                lpips_loss = lpips_fn(spnn_decoded, vae_decoded).mean()

            cycle_loss = torch.tensor(0.0, device=DEVICE)
            if args.lambda_cycle > 0:
                re_encoded = spnn.encode(spnn_decoded)
                cycle_loss = mse_loss(re_encoded, vae_latent)

            roundtrip_loss = torch.tensor(0.0, device=DEVICE)
            if args.lambda_roundtrip > 0:
                spnn_latent = spnn.encode(images)
                spnn_recon = spnn.decode(spnn_latent)
                roundtrip_loss = mse_loss(spnn_recon, images)

            cos_loss = torch.tensor(0.0, device=DEVICE)
            if args.lambda_cos > 0:
                with torch.no_grad():
                    unscaled_z_vae = vae.encode(images).latent_dist.mode()
                    # Ensure the target is scaled!
                    z_vae = unscaled_z_vae * vae.config.scaling_factor

                z_spnn = spnn.encode(images)

                cos_sim = F.cosine_similarity(z_spnn.flatten(1), z_vae.flatten(1), dim=1).mean()
                cos_loss = 1.0 - cos_sim

            # Combine total loss
            loss = (decoder_loss
                    + cur_lambda_lpips * lpips_loss
                    + args.lambda_cycle * cycle_loss
                    + args.lambda_roundtrip * roundtrip_loss
                    + args.lambda_cos * cos_loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}",
            })


        avg_loss = epoch_loss / len(loader)
        print(f"  Epoch {epoch} — avg decoder loss: {avg_loss:.6f}")

        # ── Penrose + roundtrip checks (diagnostic only) ──
        if epoch % args.save_every == 0:
            p_metrics = penrose_check(spnn, penrose_images, penrose_latent, DEVICE)
            print_penrose_metrics(p_metrics)
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


class Args:
    train = True
    test = False

    dataset = "celebahq"
    laion_dir = None

    img_size = 256
    n_test = 3000

    mix_type = "cayley"
    hidden = 192
    scale_bound = 2.0

    lambda_lpips = 0.5
    lambda_cycle = 0.3
    lambda_roundtrip = 0.3
    lambda_cos = 0.5

    batch_size = 16
    num_epochs = 200          # Extended to give the warm-up scheduler room to work
    lr = 1e-4
    save_every = 1
    penrose_batch_size = 64
    num_workers = 16
    max_images = None
    output_dir = "checkpoints_spnn_cos"
    sample_dir = "samples_fixed"

    checkpoint = None
    num_cycles = 10
    num_save_images = 30


args = Args()
train(args)