"""
SD img2img cycle comparison: VAE vs SPNN.

Uses StableDiffusionImg2ImgPipeline from diffusers with VAE swapped for SPNN.
"""

import argparse
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
from diffusers import StableDiffusionImg2ImgPipeline, DDIMScheduler
import wandb
from models import SPNNAutoencoder
from dataset import CelebAHQDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class SPNNVAE(nn.Module):
    def __init__(self, spnn, original_vae):
        super().__init__()
        self.spnn = spnn
        self.config = original_vae.config

    @property
    def device(self): return next(self.spnn.parameters()).device

    @property
    def dtype(self): return next(self.spnn.parameters()).dtype

    def encode(self, x):
        z = self.spnn.encode(x)

        class DummyDist:
            def mode(self): return z
            def sample(self, generator=None): return z

        class DummyOutput:
            def __init__(self, dist): self.latent_dist = dist

        return DummyOutput(DummyDist())

    def decode(self, z, return_dict=True, **kwargs):
        image = self.spnn.decode(z)

        if not return_dict:
            return (image,)

        class DummyOutput:
            def __init__(self, sample): self.sample = sample

        return DummyOutput(image)


def calc_psnr(img1, img2):
    """Compute PSNR between two [0,1] tensors."""
    mse = F.mse_loss(img1, img2).item()
    if mse == 0:
        return float("inf")
    return 10 * math.log10(1.0 / mse)


def main():
    parser = argparse.ArgumentParser(description="SD img2img cycles: VAE vs SPNN")
    parser.add_argument("--num_images", type=int, default=100)
    parser.add_argument("--num_cycles", type=int, default=10)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--strength", type=float, default=0.5,
                        help="Noise strength (0=no change, 1=full denoise)")
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints_celebahq/spnn_vae_best.pt")
    parser.add_argument("--num_save_grids", type=int, default=5)
    args = parser.parse_args()

    num_images = args.num_images
    num_cycles = args.num_cycles

    wandb.init(project="spnn-vae",
               name=f"img2img_cycles_N{num_images}_s{args.strength}",
               config=vars(args))

    sd_id = "runwayml/stable-diffusion-v1-5"

    # VAE pipeline (use DDIM scheduler for deterministic, stable results)
    pipe_vae = StableDiffusionImg2ImgPipeline.from_pretrained(
        sd_id, torch_dtype=torch.float32, safety_checker=None,
    ).to(DEVICE)
    pipe_vae.scheduler = DDIMScheduler.from_config(pipe_vae.scheduler.config)

    # SPNN pipeline
    pipe_spnn = StableDiffusionImg2ImgPipeline.from_pretrained(
        sd_id, torch_dtype=torch.float32, safety_checker=None,
    ).to(DEVICE)
    pipe_spnn.scheduler = DDIMScheduler.from_config(pipe_spnn.scheduler.config)
    spnn = SPNNAutoencoder(mix_type='cayley', hidden=256, r_hidden=256,
                           scale_bound=2.0).to(DEVICE)
    ckpt = torch.load(args.checkpoint, map_location=DEVICE)
    spnn.load_state_dict(ckpt.get("model_state_dict", ckpt))
    spnn.eval()
    pipe_spnn.vae = SPNNVAE(spnn, pipe_spnn.vae)

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
        img_tensor = test_dataset[img_idx]  # [-1, 1]
        # Convert to [0, 1] tensor for pipeline input
        original_tensor = (img_tensor + 1) / 2  # [0, 1]

        # Pipeline accepts tensors directly when using output_type="pt"
        curr_vae_tensor = original_tensor.unsqueeze(0).to(DEVICE)  # [1, 3, H, W]
        curr_spnn_tensor = original_tensor.unsqueeze(0).to(DEVICE)
        prev_vae_tensor = original_tensor
        prev_spnn_tensor = original_tensor
        vae_all_tensors = [original_tensor]
        spnn_all_tensors = [original_tensor]

        for c in range(num_cycles):
            seed = 42 + c

            curr_vae_tensor = pipe_vae(
                prompt="",
                image=curr_vae_tensor,
                strength=args.strength,
                num_inference_steps=args.num_inference_steps,
                generator=torch.Generator(device=DEVICE).manual_seed(seed),
                output_type="pt",
            ).images  # [1, 3, H, W] in [0, 1]

            curr_spnn_tensor = pipe_spnn(
                prompt="",
                image=curr_spnn_tensor,
                strength=args.strength,
                num_inference_steps=args.num_inference_steps,
                generator=torch.Generator(device=DEVICE).manual_seed(seed),
                output_type="pt",
            ).images  # [1, 3, H, W] in [0, 1]

            vae_tensor_cur = curr_vae_tensor[0].cpu()
            spnn_tensor_cur = curr_spnn_tensor[0].cpu()

            vae_psnr_t = calc_psnr(vae_tensor_cur, original_tensor)
            spnn_psnr_t = calc_psnr(spnn_tensor_cur, original_tensor)
            vae_psnr_s = calc_psnr(vae_tensor_cur, prev_vae_tensor)
            spnn_psnr_s = calc_psnr(spnn_tensor_cur, prev_spnn_tensor)

            vae_psnr_total_sum[c] += vae_psnr_t
            spnn_psnr_total_sum[c] += spnn_psnr_t
            vae_psnr_step_sum[c] += vae_psnr_s
            spnn_psnr_step_sum[c] += spnn_psnr_s
            vae_psnr_total_sq[c] += vae_psnr_t ** 2
            spnn_psnr_total_sq[c] += spnn_psnr_t ** 2

            prev_vae_tensor = vae_tensor_cur
            prev_spnn_tensor = spnn_tensor_cur
            vae_all_tensors.append(vae_tensor_cur)
            spnn_all_tensors.append(spnn_tensor_cur)

        n_done = img_idx + 1
        print(f"[{n_done}/{num_images}] "
              f"VAE total@{num_cycles}={vae_psnr_total_sum[-1]/n_done:.2f}dB  "
              f"SPNN total@{num_cycles}={spnn_psnr_total_sum[-1]/n_done:.2f}dB")

        # Save full cycle grids
        if img_idx < args.num_save_grids:
            vae_row = torch.stack(vae_all_tensors)
            spnn_row = torch.stack(spnn_all_tensors)
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
