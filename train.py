import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from diffusers import AutoencoderKL
from models import SPNNAutoencoder
from diagnostics import penrose_check, print_penrose_metrics

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    parser = argparse.ArgumentParser(description="Train SPNN VAE decoder via distillation")
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--sample_dir", type=str, default="samples")
    return parser.parse_args()


class CelebAHQDataset(Dataset):
    def __init__(self, img_size=256, max_images=None):
        from datasets import load_dataset
        print("Loading Ryan-sjtu/celebahq-caption dataset...")
        ds = load_dataset("Ryan-sjtu/celebahq-caption", split="train")
        if max_images is not None:
            ds = ds.select(range(min(max_images, len(ds))))
        self.ds = ds
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        img = item["image"]
        if not isinstance(img, Image.Image):
            img = Image.open(img)
        img = img.convert("RGB")
        return self.transform(img)


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
    os.makedirs(args.sample_dir, exist_ok=True)

    dataset = CelebAHQDataset(img_size=args.img_size, max_images=args.max_images)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True
    )
    print(f"Dataset: {len(dataset)} images, {len(loader)} batches/epoch")

    # ── Models ──
    vae = load_sd_vae()
    spnn = SPNNAutoencoder(mix_type="cayley", hidden=128, scale_bound=2.0).to(DEVICE)

    total_params = sum(p.numel() for p in spnn.parameters())
    print(f"SPNN total params: {total_params:,}")

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

            loss = mse_loss(spnn_decoded, vae_decoded)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(spnn.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}",
            })

            if batch_idx % 200 == 0:
                save_comparison(spnn_decoded, vae_decoded, images, epoch, batch_idx, args.sample_dir)

        avg_loss = epoch_loss / len(loader)
        print(f"  Epoch {epoch} — avg decoder loss: {avg_loss:.6f}")

        # ── Penrose + roundtrip checks (diagnostic only) ──
        if epoch % args.save_every == 0:
            p_metrics = penrose_check(spnn, images, vae_latent, DEVICE)
            print_penrose_metrics(p_metrics)
            # TODO: wandb.log(p_metrics, step=epoch) when wandb is integrated
            spnn.train()

            ckpt_path = os.path.join(args.output_dir, f"spnn_epoch{epoch:03d}.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": spnn.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
            }, ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}")

    # ── Final save ──
    final_path = os.path.join(args.output_dir, "spnn_final.pt")
    torch.save(spnn.state_dict(), final_path)
    print(f"\nTraining complete. Final model: {final_path}")
    print(f"The encoder (spnn.encode / forward) now works automatically —")
    print(f"it uses the same s, t, mix that were trained through the decoder.")


if __name__ == "__main__":
    args = parse_args()
    train(args)