"""
Manual img2img comparison: VAE vs SPNN.

Pipeline: encode → add noise → UNet denoise → decode (no text guidance)
Uses the UNet + scheduler from SD 1.5 directly, bypassing the pipeline.
"""

import argparse
import math
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm import tqdm
import wandb
from models import SPNNAutoencoder
from dataset import CelebAHQDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def calc_psnr(img1, img2):
    """Compute PSNR between two [-1,1] tensors."""
    mse = F.mse_loss(img1, img2).item()
    if mse == 0:
        return float("inf")
    return 10 * math.log10(4.0 / mse)  # range=2 for [-1,1]


@torch.no_grad()
def img2img_manual(unet, scheduler, encode_fn, decode_fn, scaling_factor,
                   image, empty_emb, strength, num_steps):
    """
    Manual img2img: encode → add noise → denoise → decode.

    Args:
        encode_fn: callable, image [-1,1] → latent (unscaled)
        decode_fn: callable, latent (unscaled) → image [-1,1]
        scaling_factor: VAE scaling factor (0.18215)
        image: [B, 3, H, W] in [-1, 1]
        empty_emb: [1, 77, 768] unconditional text embedding
        strength: fraction of noise to add (0=none, 1=full noise)
        num_steps: total DDIM steps
    """
    scheduler.set_timesteps(num_steps)

    # Encode
    z = encode_fn(image) * scaling_factor

    # Add noise — skip first (1-strength) fraction of timesteps
    start_step = int(num_steps * (1 - strength))
    t_start = scheduler.timesteps[start_step]
    noise = torch.randn_like(z)
    z_noisy = scheduler.add_noise(z, noise, t_start)

    # Denoise from t_start
    emb = empty_emb.expand(z.shape[0], -1, -1)
    z_t = z_noisy
    for t in scheduler.timesteps[start_step:]:
        noise_pred = unet(z_t, t, encoder_hidden_states=emb).sample
        z_t = scheduler.step(noise_pred, t, z_t).prev_sample

    # Decode
    result = decode_fn(z_t / scaling_factor)
    return result.clamp(-1, 1)


def main():
    parser = argparse.ArgumentParser(description="Manual img2img: VAE vs SPNN")
    parser.add_argument("--num_images", type=int, default=100)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--strength", type=float, default=0.5,
                        help="Noise strength (0=no change, 1=full denoise)")
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints_celebahq/spnn_vae_best.pt")
    parser.add_argument("--num_save_grids", type=int, default=5)
    args = parser.parse_args()

    num_images = args.num_images

    wandb.init(project="spnn-vae",
               name=f"img2img_manual_N{num_images}_s{args.strength}",
               config=vars(args))

    # Load models
    sd_id = "runwayml/stable-diffusion-v1-5"

    print("Loading UNet...")
    unet = UNet2DConditionModel.from_pretrained(sd_id, subfolder="unet")
    unet.eval().to(DEVICE)

    scheduler = DDIMScheduler.from_pretrained(sd_id, subfolder="scheduler")

    print("Loading text encoder (for unconditional embedding)...")
    tokenizer = CLIPTokenizer.from_pretrained(sd_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(sd_id, subfolder="text_encoder")
    text_encoder.eval().to(DEVICE)
    tokens = tokenizer("", padding="max_length", max_length=77,
                       return_tensors="pt").input_ids.to(DEVICE)
    with torch.no_grad():
        empty_emb = text_encoder(tokens).last_hidden_state  # [1, 77, 768]
    del text_encoder, tokenizer  # free memory

    print("Loading VAE...")
    vae = AutoencoderKL.from_pretrained(sd_id, subfolder="vae")
    vae.eval().to(DEVICE)
    scaling_factor = vae.config.scaling_factor

    print(f"Loading SPNN from {args.checkpoint}...")
    spnn = SPNNAutoencoder(mix_type='cayley', hidden=256, r_hidden=256,
                           scale_bound=2.0).to(DEVICE)
    ckpt = torch.load(args.checkpoint, map_location=DEVICE)
    spnn.load_state_dict(ckpt.get("model_state_dict", ckpt))
    spnn.eval()

    # Codec functions
    def vae_encode(x): return vae.encode(x).latent_dist.mode()
    def vae_decode(z): return vae.decode(z).sample
    def spnn_encode(x): return spnn.encode(x)
    def spnn_decode(z): return spnn.decode(z)

    # Dataset
    test_dataset = CelebAHQDataset(img_size=512, split="test", n_test=1000)
    num_images = min(num_images, len(test_dataset))
    print(f"Running single img2img pass over {num_images} test images "
          f"(strength={args.strength}, steps={args.num_inference_steps})")

    vae_psnrs = []
    spnn_psnrs = []

    for img_idx in tqdm(range(num_images), desc="Images"):
        x = test_dataset[img_idx].unsqueeze(0).to(DEVICE)  # [1, 3, H, W] in [-1, 1]

        result_vae = img2img_manual(
            unet, scheduler, vae_encode, vae_decode, scaling_factor,
            x, empty_emb, args.strength, args.num_inference_steps,
        )
        result_spnn = img2img_manual(
            unet, scheduler, spnn_encode, spnn_decode, scaling_factor,
            x, empty_emb, args.strength, args.num_inference_steps,
        )

        vae_psnr = calc_psnr(result_vae, x)
        spnn_psnr = calc_psnr(result_spnn, x)
        vae_psnrs.append(vae_psnr)
        spnn_psnrs.append(spnn_psnr)

        if img_idx < args.num_save_grids:
            grid = torch.stack([
                (x[0].cpu() + 1) / 2,
                (result_vae[0].cpu() + 1) / 2,
                (result_spnn[0].cpu() + 1) / 2,
            ])
            save_image(grid, f"img2img_grid_{img_idx:03d}.png",
                       nrow=3, padding=2, pad_value=1.0)

    # Log summary
    mean_vae = np.mean(vae_psnrs)
    mean_spnn = np.mean(spnn_psnrs)
    std_vae = np.std(vae_psnrs)
    std_spnn = np.std(spnn_psnrs)

    print(f"\n{'='*50}")
    print(f"Results over {num_images} images:")
    print(f"  VAE:  {mean_vae:.2f} +/- {std_vae:.2f} dB")
    print(f"  SPNN: {mean_spnn:.2f} +/- {std_spnn:.2f} dB")
    print(f"  Gap:  {mean_spnn - mean_vae:+.2f} dB")

    wandb.log({
        "mean_vae_psnr": mean_vae,
        "mean_spnn_psnr": mean_spnn,
        "std_vae_psnr": std_vae,
        "std_spnn_psnr": std_spnn,
        "mean_psnr_gap": mean_spnn - mean_vae,
    })

    wandb.finish()
    print("Done.")


if __name__ == "__main__":
    main()
