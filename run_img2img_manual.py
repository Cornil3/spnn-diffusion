"""
Manual img2img cycle comparison: VAE vs SPNN.

Pipeline per cycle: encode → add noise → UNet denoise → decode (no text guidance)
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
                   image, empty_emb, strength, num_steps, generator=None):
    """
    Manual img2img: encode → add noise → denoise → decode.
    """
    scheduler.set_timesteps(num_steps)

    # Encode
    z = encode_fn(image) * scaling_factor

    # Add noise — skip first (1-strength) fraction of timesteps
    start_step = int(num_steps * (1 - strength))
    t_start = scheduler.timesteps[start_step]
    noise = torch.randn(z.shape, device=z.device, dtype=z.dtype, generator=generator)
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
    parser = argparse.ArgumentParser(description="Manual img2img cycles: VAE vs SPNN")
    parser.add_argument("--num_images", type=int, default=1)
    parser.add_argument("--num_cycles", type=int, default=10)
    parser.add_argument("--num_inference_steps", type=int, default=100)
    parser.add_argument("--strength", type=float, default=0.2,
                        help="Noise strength (0=no change, 1=full denoise)")
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints_celebahq/spnn_vae_best.pt")
    parser.add_argument("--num_save_grids", type=int, default=1)
    args = parser.parse_args()

    num_images = args.num_images
    num_cycles = args.num_cycles

    wandb.init(project="spnn-vae",
               name=f"img2img_cycles_N{num_images}_s{args.strength}",
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
    print(f"Running {num_cycles} cycles over {num_images} test images "
          f"(strength={args.strength}, steps={args.num_inference_steps})")

    # Accumulators
    vae_psnr_total_sum = np.zeros(num_cycles)
    spnn_psnr_total_sum = np.zeros(num_cycles)
    vae_psnr_step_sum = np.zeros(num_cycles)
    spnn_psnr_step_sum = np.zeros(num_cycles)
    vae_psnr_total_sq = np.zeros(num_cycles)
    spnn_psnr_total_sq = np.zeros(num_cycles)

    for img_idx in range(num_images):
        x_orig = test_dataset[img_idx].unsqueeze(0).to(DEVICE)  # [1, 3, H, W] in [-1, 1]

        x_vae = x_orig.clone()
        x_spnn = x_orig.clone()
        vae_all = [(x_orig[0].cpu() + 1) / 2]
        spnn_all = [(x_orig[0].cpu() + 1) / 2]

        for c in range(num_cycles):
            seed = 42 + c

            x_vae = img2img_manual(
                unet, scheduler, vae_encode, vae_decode, scaling_factor,
                x_vae, empty_emb, args.strength, args.num_inference_steps,
                generator=torch.Generator(device=DEVICE).manual_seed(seed),
            )

            x_spnn = img2img_manual(
                unet, scheduler, spnn_encode, spnn_decode, scaling_factor,
                x_spnn, empty_emb, args.strength, args.num_inference_steps,
                generator=torch.Generator(device=DEVICE).manual_seed(seed),
            )

            vae_psnr_t = calc_psnr(x_vae, x_orig)
            spnn_psnr_t = calc_psnr(x_spnn, x_orig)
            vae_psnr_s = calc_psnr(x_vae, x_orig if c == 0 else prev_vae)
            spnn_psnr_s = calc_psnr(x_spnn, x_orig if c == 0 else prev_spnn)

            vae_psnr_total_sum[c] += vae_psnr_t
            spnn_psnr_total_sum[c] += spnn_psnr_t
            vae_psnr_step_sum[c] += vae_psnr_s
            spnn_psnr_step_sum[c] += spnn_psnr_s
            vae_psnr_total_sq[c] += vae_psnr_t ** 2
            spnn_psnr_total_sq[c] += spnn_psnr_t ** 2

            prev_vae = x_vae.clone()
            prev_spnn = x_spnn.clone()
            vae_all.append((x_vae[0].cpu() + 1) / 2)
            spnn_all.append((x_spnn[0].cpu() + 1) / 2)

        n_done = img_idx + 1
        print(f"[{n_done}/{num_images}] "
              f"VAE total@{num_cycles}={vae_psnr_total_sum[-1]/n_done:.2f}dB  "
              f"SPNN total@{num_cycles}={spnn_psnr_total_sum[-1]/n_done:.2f}dB")

        # Save full cycle grids
        if img_idx < args.num_save_grids:
            vae_row = torch.stack(vae_all)
            spnn_row = torch.stack(spnn_all)
            grid = torch.cat([vae_row, spnn_row], dim=0)
            grid_path = f"img2img_grid_{img_idx:03d}.png"
            save_image(grid, grid_path,
                       nrow=num_cycles + 1, padding=2, pad_value=1.0)
            wandb.log({f"cycle_grid_img{img_idx}": wandb.Image(
                grid_path, caption=f"Top: VAE, Bottom: SPNN (img {img_idx})")})

    # Log mean PSNR per cycle
    for c in range(num_cycles):
        n = num_images
        mean_vae = vae_psnr_total_sum[c] / n
        mean_spnn = spnn_psnr_total_sum[c] / n
        std_vae = np.sqrt(max(0, vae_psnr_total_sq[c] / n - mean_vae ** 2))
        std_spnn = np.sqrt(max(0, spnn_psnr_total_sq[c] / n - mean_spnn ** 2))
        mean_vae_step = vae_psnr_step_sum[c] / n
        mean_spnn_step = spnn_psnr_step_sum[c] / n

        print(f"Cycle {c+1}: VAE={mean_vae:.2f}+/-{std_vae:.2f}dB  "
              f"SPNN={mean_spnn:.2f}+/-{std_spnn:.2f}dB  "
              f"gap={mean_spnn - mean_vae:+.2f}dB")

        wandb.log({
            "cycle": c + 1,
            "mean_vae_psnr_vs_original": mean_vae,
            "mean_spnn_psnr_vs_original": mean_spnn,
            "std_vae_psnr_vs_original": std_vae,
            "std_spnn_psnr_vs_original": std_spnn,
            "mean_psnr_gap_vs_original": mean_spnn - mean_vae,
            "mean_vae_psnr_vs_prev": mean_vae_step,
            "mean_spnn_psnr_vs_prev": mean_spnn_step,
            "mean_psnr_gap_vs_prev": mean_spnn_step - mean_vae_step,
        })

    wandb.finish()
    print("Done.")


if __name__ == "__main__":
    main()
