"""
SD Diffusion decode comparison: VAE vs SPNN.

Two modes:
  txt2img — full text-to-image generation, decode final latent with both codecs
  img2img — encode real image, add noise, denoise, decode with both codecs

This is a single-decode test (no iterative encode/decode like DDNM).
"""

import argparse
import math
import os

import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from torchvision.utils import save_image
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from dataset import CelebAHQDataset
from models import SPNNAutoencoder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ═══════════════════════════════════════════════════════════
# Model loading
# ═══════════════════════════════════════════════════════════

def load_models(args):
    """Load SD UNet, scheduler, text encoder, tokenizer, VAE, and SPNN."""
    sd_id = "runwayml/stable-diffusion-v1-5"

    print("Loading UNet...")
    unet = UNet2DConditionModel.from_pretrained(sd_id, subfolder="unet")
    unet.eval().to(DEVICE)
    for p in unet.parameters():
        p.requires_grad = False

    print("Loading scheduler...")
    scheduler = DDIMScheduler.from_pretrained(sd_id, subfolder="scheduler")
    scheduler.set_timesteps(args.num_steps)

    print("Loading text encoder...")
    tokenizer = CLIPTokenizer.from_pretrained(sd_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(sd_id, subfolder="text_encoder")
    text_encoder.eval().to(DEVICE)
    for p in text_encoder.parameters():
        p.requires_grad = False

    vae_id = "timbrooks/instruct-pix2pix"
    print("Loading VAE (sd-vae-ft-mse via pix2pix — compatible with SD v1.5)...")
    vae = AutoencoderKL.from_pretrained(vae_id, subfolder="vae")
    vae.eval().to(DEVICE)
    for p in vae.parameters():
        p.requires_grad = False

    print(f"Loading SPNN from {args.checkpoint}...")
    spnn = SPNNAutoencoder(
        mix_type=args.mix_type, hidden=args.hidden, scale_bound=args.scale_bound,
    )
    state = torch.load(args.checkpoint, map_location=DEVICE, weights_only=True)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    spnn.load_state_dict(state)
    spnn.eval().to(DEVICE)

    return unet, scheduler, tokenizer, text_encoder, vae, spnn


# ═══════════════════════════════════════════════════════════
# Codec wrappers
# ═══════════════════════════════════════════════════════════

class VAECodec:
    def __init__(self, vae):
        self.vae = vae
        self.sf = vae.config.scaling_factor

    def encode(self, x):
        return self.vae.encode(x).latent_dist.mode() * self.sf

    def decode(self, z):
        return self.vae.decode(z / self.sf).sample


class SPNNCodec:
    def __init__(self, spnn):
        self.spnn = spnn

    def encode(self, x):
        return self.spnn.encode(x)

    def decode(self, z):
        return self.spnn.decode(z)


# ═══════════════════════════════════════════════════════════
# Text encoding helper
# ═══════════════════════════════════════════════════════════

@torch.no_grad()
def encode_text(tokenizer, text_encoder, prompt):
    """Encode a text prompt into CLIP embeddings."""
    tokens = tokenizer(
        prompt, padding="max_length", max_length=77,
        truncation=True, return_tensors="pt",
    ).input_ids.to(DEVICE)
    return text_encoder(tokens).last_hidden_state  # [1, 77, 768]


# ═══════════════════════════════════════════════════════════
# DDIM reverse process (text-conditioned generation)
# ═══════════════════════════════════════════════════════════

@torch.no_grad()
def ddim_sample(unet, scheduler, text_emb, uncond_emb, guidance_scale,
                latent_shape, start_latent=None, start_step=0):
    """
    Run DDIM reverse process.

    Args:
        start_latent: if provided, start from this latent instead of pure noise
        start_step: index into scheduler.timesteps to start from (0 = full denoise)
    Returns:
        Final clean latent z_0
    """
    alphas_cumprod = scheduler.alphas_cumprod.to(DEVICE)
    timesteps = scheduler.timesteps

    if start_latent is not None:
        z_t = start_latent
    else:
        z_t = torch.randn(latent_shape, device=DEVICE)

    B = z_t.shape[0]
    emb = text_emb.expand(B, -1, -1)
    uncond = uncond_emb.expand(B, -1, -1)

    for i, t in enumerate(tqdm(timesteps[start_step:], desc="DDIM steps", leave=False)):
        step_idx = start_step + i
        t_tensor = t.unsqueeze(0).expand(B).to(DEVICE)

        # CFG
        if guidance_scale > 1.0:
            z_in = torch.cat([z_t, z_t], dim=0)
            t_in = t_tensor.repeat(2)
            e_in = torch.cat([uncond, emb], dim=0)
            noise_pred_all = unet(z_in, t_in, encoder_hidden_states=e_in).sample
            noise_uncond, noise_cond = noise_pred_all.chunk(2)
            noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
        else:
            noise_pred = unet(z_t, t_tensor, encoder_hidden_states=emb).sample

        # Estimate clean latent
        alpha_t = alphas_cumprod[t].view(1, 1, 1, 1)
        z_0_pred = (z_t - (1 - alpha_t).sqrt() * noise_pred) / alpha_t.sqrt()

        # DDIM step to z_{t-1}
        if step_idx < len(timesteps) - 1:
            t_next = timesteps[step_idx + 1]
            alpha_next = alphas_cumprod[t_next].view(1, 1, 1, 1)
        else:
            alpha_next = torch.tensor(1.0, device=DEVICE).view(1, 1, 1, 1)

        z_t = alpha_next.sqrt() * z_0_pred + (1 - alpha_next).sqrt() * noise_pred

    return z_t


# ═══════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════

def psnr(pred, target):
    """PSNR in dB, images in [-1, 1]."""
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return float("inf")
    return 10 * math.log10(4.0 / mse.item())


def compute_lpips(img1, img2):
    """Compute LPIPS between two images. Lazy-loads the model."""
    if not hasattr(compute_lpips, "_model"):
        import lpips
        compute_lpips._model = lpips.LPIPS(net="alex").to(DEVICE)
    with torch.no_grad():
        return compute_lpips._model(img1, img2).item()


# ═══════════════════════════════════════════════════════════
# Mode 1: Text-to-image
# ═══════════════════════════════════════════════════════════

@torch.no_grad()
def run_txt2img(args, unet, scheduler, tokenizer, text_encoder, vae_codec, spnn_codec):
    """Generate images from text prompts, decode with both VAE and SPNN."""
    out_dir = os.path.join("decode_compare", "txt2img")
    os.makedirs(out_dir, exist_ok=True)

    # Empty text embedding for CFG
    uncond_emb = encode_text(tokenizer, text_encoder, "")

    latent_shape = (1, 4, args.img_size // 8, args.img_size // 8)
    lpips_scores = []

    for idx, prompt in enumerate(args.prompts):
        print(f"\n  [{idx+1}/{len(args.prompts)}] \"{prompt}\"")

        # Encode text
        text_emb = encode_text(tokenizer, text_encoder, prompt)

        # Generate latent via full DDIM reverse
        z_0 = ddim_sample(
            unet, scheduler, text_emb, uncond_emb, args.guidance_scale,
            latent_shape,
        )

        # Decode with both codecs
        img_vae = vae_codec.decode(z_0).clamp(-1, 1)
        img_spnn = spnn_codec.decode(z_0).clamp(-1, 1)

        # LPIPS between the two decodes
        lp = compute_lpips(img_vae, img_spnn)
        lpips_scores.append(lp)
        print(f"    LPIPS(VAE, SPNN): {lp:.4f}")

        # Save grid: VAE | SPNN
        grid = torch.stack([
            (img_vae[0].cpu() + 1) / 2,
            (img_spnn[0].cpu() + 1) / 2,
        ])
        save_image(grid, os.path.join(out_dir, f"prompt_{idx:03d}.png"),
                   nrow=2, padding=2)

    avg_lpips = sum(lpips_scores) / len(lpips_scores)
    print(f"\n  txt2img avg LPIPS(VAE, SPNN): {avg_lpips:.4f}")


# ═══════════════════════════════════════════════════════════
# Mode 2: Img2img
# ═══════════════════════════════════════════════════════════

@torch.no_grad()
def run_img2img(args, unet, scheduler, tokenizer, text_encoder, vae_codec, spnn_codec):
    """Encode real images, add noise, denoise, decode with both codecs."""
    out_dir = os.path.join("decode_compare", "img2img")
    os.makedirs(out_dir, exist_ok=True)

    # Load test images
    dataset = CelebAHQDataset(img_size=args.img_size, split="test", n_test=1000)
    images = torch.stack([dataset[i] for i in range(args.num_images)]).to(DEVICE)
    print(f"  Loaded {images.shape[0]} test images ({images.shape[1:]})")

    # Empty text for unconditional denoising + CFG
    uncond_emb = encode_text(tokenizer, text_encoder, "")
    text_emb = uncond_emb  # unconditional img2img

    alphas_cumprod = scheduler.alphas_cumprod.to(DEVICE)
    timesteps = scheduler.timesteps

    # Find the starting timestep based on noise_strength
    start_step = max(0, len(timesteps) - int(len(timesteps) * args.noise_strength))
    t_start = timesteps[start_step]
    alpha_t = alphas_cumprod[t_start].view(1, 1, 1, 1)
    print(f"  noise_strength={args.noise_strength}, start_step={start_step}, "
          f"t_start={t_start.item()}, alpha_t={alpha_t.item():.4f}")

    metrics = {"VAE": {"psnr": [], "mse": []}, "SPNN": {"psnr": [], "mse": []}}

    for idx in range(images.shape[0]):
        x_gt = images[idx:idx+1]  # [1, 3, H, W]
        print(f"\n  [{idx+1}/{images.shape[0]}]")

        # Encode with VAE
        z_0 = vae_codec.encode(x_gt)

        # Add noise at timestep t_start
        noise = torch.randn_like(z_0)
        z_t = alpha_t.sqrt() * z_0 + (1 - alpha_t).sqrt() * noise

        # Denoise from t_start back to 0
        z_clean = ddim_sample(
            unet, scheduler, text_emb, uncond_emb, args.guidance_scale,
            z_0.shape, start_latent=z_t, start_step=start_step,
        )

        # Decode with both codecs
        results = {}
        for name, codec in [("VAE", vae_codec), ("SPNN", spnn_codec)]:
            img = codec.decode(z_clean).clamp(-1, 1)
            results[name] = img

            p = psnr(img, x_gt)
            m = F.mse_loss(img, x_gt).item()
            metrics[name]["psnr"].append(p)
            metrics[name]["mse"].append(m)
            print(f"    {name}: PSNR={p:.2f} dB, MSE={m:.6f}")

        # Save grid: Original | VAE | SPNN
        grid = torch.stack([
            (x_gt[0].cpu() + 1) / 2,
            (results["VAE"][0].cpu() + 1) / 2,
            (results["SPNN"][0].cpu() + 1) / 2,
        ])
        save_image(grid, os.path.join(out_dir, f"img{idx:03d}.png"),
                   nrow=3, padding=2)

    # Summary
    print(f"\n  img2img Summary:")
    for name in ["VAE", "SPNN"]:
        avg_psnr = sum(metrics[name]["psnr"]) / len(metrics[name]["psnr"])
        avg_mse = sum(metrics[name]["mse"]) / len(metrics[name]["mse"])
        print(f"    {name:>5s}: avg PSNR = {avg_psnr:.2f} dB, avg MSE = {avg_mse:.6f}")


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

def run(args):
    torch.manual_seed(args.seed)

    unet, scheduler, tokenizer, text_encoder, vae, spnn = load_models(args)
    vae_codec = VAECodec(vae)
    spnn_codec = SPNNCodec(spnn)

    if args.mode in ("txt2img", "both"):
        print(f"\n{'=' * 60}")
        print("Mode: txt2img")
        print(f"{'=' * 60}")
        run_txt2img(args, unet, scheduler, tokenizer, text_encoder,
                    vae_codec, spnn_codec)

    if args.mode in ("img2img", "both"):
        print(f"\n{'=' * 60}")
        print("Mode: img2img")
        print(f"{'=' * 60}")
        run_img2img(args, unet, scheduler, tokenizer, text_encoder,
                    vae_codec, spnn_codec)

    print("\nDone. Results saved to decode_compare/")


def parse_args():
    parser = argparse.ArgumentParser(
        description="SD diffusion decode comparison: VAE vs SPNN")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="SPNN model checkpoint path")
    parser.add_argument("--mode", type=str, default="both",
                        choices=["txt2img", "img2img", "both"],
                        help="Which mode(s) to run")
    parser.add_argument("--prompts", nargs="+",
                        default=["a photo of a person",
                                 "a portrait of a woman smiling"],
                        help="Text prompts for txt2img mode")
    parser.add_argument("--num_images", type=int, default=5,
                        help="Number of test images for img2img")
    parser.add_argument("--noise_strength", type=float, default=0.5,
                        help="Noise fraction for img2img (0-1)")
    parser.add_argument("--num_steps", type=int, default=50,
                        help="DDIM steps")
    parser.add_argument("--guidance_scale", type=float, default=7.5,
                        help="Classifier-free guidance scale")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--img_size", type=int, default=256)
    # Model args (must match checkpoint)
    parser.add_argument("--mix_type", type=str, default="cayley",
                        choices=["cayley", "householder"])
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--scale_bound", type=float, default=2.0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)
