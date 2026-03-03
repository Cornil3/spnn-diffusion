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

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 256
BATCH_SIZE = 4
NUM_EPOCHS = 1
LR = 1e-4
SAVE_EVERY = 5
NUM_WORKERS = 2
MAX_IMAGES = None
OUTPUT_DIR = "checkpoints"
SAMPLE_DIR = "samples"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SAMPLE_DIR, exist_ok=True)


class CelebAHQDataset(Dataset):
    def __init__(self, max_images=None):
        from datasets import load_dataset
        print("Loading Ryan-sjtu/celebahq-caption dataset...")
        ds = load_dataset("Ryan-sjtu/celebahq-caption", split="train")
        if max_images is not None:
            ds = ds.select(range(min(max_images, len(ds))))
        self.ds = ds
        self.transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
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

def save_comparison(spnn_decoded, vae_decoded, original, epoch, batch_idx):
    from torchvision.utils import save_image
    n = min(4, original.size(0))
    orig = (original[:n].cpu() + 1) / 2
    vae_r = (vae_decoded[:n].cpu() + 1) / 2
    spnn_r = (spnn_decoded[:n].detach().cpu() + 1) / 2
    grid = torch.cat([orig, vae_r, spnn_r], dim=0)
    path = os.path.join(SAMPLE_DIR, f"epoch{epoch:03d}_batch{batch_idx:04d}.png")
    save_image(grid, path, nrow=n, padding=2)

def train():
    print(f"Device: {DEVICE}")

    dataset = CelebAHQDataset(max_images=MAX_IMAGES)
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True
    )
    print(f"Dataset: {len(dataset)} images, {len(loader)} batches/epoch")

    # ── Models ──
    vae = load_sd_vae()
    spnn = SPNNAutoencoder(mix_type="cayley", hidden=128, scale_bound=2.0).to(DEVICE)

    total_params = sum(p.numel() for p in spnn.parameters())
    print(f"SPNN total params: {total_params:,}")

    # ── Optimizer: trains ALL of s, t, r, mix through the decoder path ──
    optimizer = torch.optim.AdamW(spnn.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS * len(loader), eta_min=1e-6
    )
    mse_loss = nn.MSELoss()

    for epoch in range(1, NUM_EPOCHS + 1):
        spnn.train()
        epoch_loss = 0.0

        pbar = tqdm(loader, desc=f"Epoch {epoch}/{NUM_EPOCHS}")
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
                save_comparison(spnn_decoded, vae_decoded, images, epoch, batch_idx)

        avg_loss = epoch_loss / len(loader)
        print(f"  Epoch {epoch} — avg decoder loss: {avg_loss:.6f}")

        # ── Sanity check: run full encode->decode and measure roundtrip ──
        if epoch % SAVE_EVERY == 0:
            spnn.eval()
            with torch.no_grad():
                sample = images[:2]
                # Encode with SPNN forward (uses the SAME s, t, mix we just trained)
                latent, side = spnn.encode(sample, return_latents=True)
                # Decode with side-info (should be near-perfect inversion)
                roundtrip = spnn.decode(latent, latents=side)
                roundtrip_err = F.mse_loss(roundtrip, sample).item()
                # Decode without side-info (uses r networks)
                roundtrip_r = spnn.decode(latent)
                roundtrip_r_err = F.mse_loss(roundtrip_r, sample).item()
                print(f"  Roundtrip error (with side-info):  {roundtrip_err:.2e}")
                print(f"  Roundtrip error (r only):          {roundtrip_r_err:.6f}")
            spnn.train()

            ckpt_path = os.path.join(OUTPUT_DIR, f"spnn_epoch{epoch:03d}.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": spnn.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
            }, ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}")

    # ── Final save ──
    final_path = os.path.join(OUTPUT_DIR, "spnn_final.pt")
    torch.save(spnn.state_dict(), final_path)
    print(f"\nTraining complete. Final model: {final_path}")
    print(f"The encoder (spnn.encode / forward) now works automatically —")
    print(f"it uses the same s, t, mix that were trained through the decoder.")