import argparse
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
from diffusers import (
    StableDiffusionInstructPix2PixPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionPipeline,
)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_images", type=int, default=1,
                        help="Number of test images to average over")
    parser.add_argument("--num_cycles", type=int, default=15)
    parser.add_argument("--num_inference_steps", type=int, default=100)
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints_celebahq_distill_512/spnn_vae_best.pt")
    parser.add_argument("--num_save_grids", type=int, default=1,
                        help="Number of example grids to save")
    parser.add_argument("--pipeline", type=str, default="pix2pix",
                        choices=["pix2pix", "sd15"],
                        help="Pipeline to use: 'pix2pix' (InstructPix2Pix) or 'sd15' (SD 1.5 img2img)")
    parser.add_argument("--strength", type=float, default=0.5,
                        help="Strength for SD 1.5 img2img (0=no change, 1=full denoise)")
    parser.add_argument("--guidance_scale", type=float, default=12,
                        help="Classifier-free guidance scale")
    parser.add_argument("--prompts", type=str, nargs="+", default=[""],
                        help="List of prompts to cycle through, e.g. "
                             "--prompts 'make it sunny' 'add glasses' 'make older'")
    parser.add_argument("--init_prompt", type=str, default=None,
                        help="If set, generate the initial image from this text prompt "
                             "(text2img) instead of loading from the dataset")
    args = parser.parse_args()

    num_cycles = args.num_cycles
    num_images = args.num_images

    use_txt2img = args.init_prompt is not None
    if use_txt2img:
        use_sd15 = True  # text2img init requires SD 1.5 for editing cycles too
    else:
        use_sd15 = args.pipeline == "sd15"
    pipeline_label = "sd15" if use_sd15 else "pix2pix"

    wandb.init(project="spnn-vae",
               name=f"image_editing_{pipeline_label}_cycles_N{num_images}",
               config=vars(args))

    if use_txt2img:
        num_images = args.num_images
        print(f"Running {num_cycles} edit cycles over {num_images} text2img seeds "
              f"(init_prompt='{args.init_prompt}', pipeline={pipeline_label})")
    else:
        test_dataset = CelebAHQDataset(img_size=512, split="test", n_test=1000)
        num_images = min(num_images, len(test_dataset))
        print(f"Running {num_cycles} cycles over {num_images} test images "
              f"(pipeline={pipeline_label}, strength={args.strength})")

    # Load SPNN model
    spnn = SPNNAutoencoder(mix_type='cayley', hidden=256, r_hidden=256, scale_bound=2.0).to(DEVICE)
    ckpt = torch.load(args.checkpoint, map_location=DEVICE)
    spnn.load_state_dict(ckpt.get("model_state_dict", ckpt))
    spnn.eval()

    # Load pipelines
    if use_sd15:
        model_id = "runwayml/stable-diffusion-v1-5"
        # Load text2img pipelines (also used for init when --init_prompt is set)
        t2i_vae = StableDiffusionPipeline.from_pretrained(
            model_id, torch_dtype=torch.float32, safety_checker=None,
        ).to(DEVICE)
        t2i_spnn = StableDiffusionPipeline.from_pretrained(
            model_id, torch_dtype=torch.float32, safety_checker=None,
        ).to(DEVICE)
        t2i_spnn.vae = SPNNVAE(spnn, t2i_spnn.vae)
        # Build img2img pipelines sharing the same components
        pipe_vae = StableDiffusionImg2ImgPipeline(
            vae=t2i_vae.vae, text_encoder=t2i_vae.text_encoder,
            tokenizer=t2i_vae.tokenizer, unet=t2i_vae.unet,
            scheduler=t2i_vae.scheduler, safety_checker=None,
            feature_extractor=t2i_vae.feature_extractor,
        )
        pipe_spnn = StableDiffusionImg2ImgPipeline(
            vae=t2i_spnn.vae, text_encoder=t2i_spnn.text_encoder,
            tokenizer=t2i_spnn.tokenizer, unet=t2i_spnn.unet,
            scheduler=t2i_spnn.scheduler, safety_checker=None,
            feature_extractor=t2i_spnn.feature_extractor,
        )
    else:
        model_id = "timbrooks/instruct-pix2pix"
        pipe_vae = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            model_id, torch_dtype=torch.float32, safety_checker=None,
        ).to(DEVICE)
        pipe_spnn = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            model_id, torch_dtype=torch.float32, safety_checker=None,
        ).to(DEVICE)
        pipe_spnn.vae = SPNNVAE(spnn, pipe_spnn.vae)

    # Accumulators: [num_cycles] arrays
    vae_psnr_total_sum = np.zeros(num_cycles)
    spnn_psnr_total_sum = np.zeros(num_cycles)
    vae_psnr_step_sum = np.zeros(num_cycles)
    spnn_psnr_step_sum = np.zeros(num_cycles)
    # For std computation
    vae_psnr_total_sq = np.zeros(num_cycles)
    spnn_psnr_total_sq = np.zeros(num_cycles)

    for img_idx in range(num_images):
        if use_txt2img:
            # Generate initial image from text with both decoders
            gen_seed = args.num_inference_steps + img_idx
            gen_kwargs = dict(
                prompt=args.init_prompt,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                output_type="pt",
            )
            gen_kwargs["generator"] = torch.Generator(device=DEVICE).manual_seed(gen_seed)
            curr_vae_tensor = t2i_vae(**gen_kwargs).images  # [1,3,H,W] in [0,1]
            gen_kwargs["generator"] = torch.Generator(device=DEVICE).manual_seed(gen_seed)
            curr_spnn_tensor = t2i_spnn(**gen_kwargs).images

            original_vae = curr_vae_tensor[0].cpu()
            original_spnn = curr_spnn_tensor[0].cpu()
            # Use VAE result as the "original" for PSNR reference
            original_tensor = original_vae
        else:
            img_tensor = test_dataset[img_idx]  # [-1, 1]
            original_tensor = (img_tensor + 1) / 2  # [0, 1]
            curr_vae_tensor = original_tensor.unsqueeze(0).to(DEVICE)
            curr_spnn_tensor = original_tensor.unsqueeze(0).to(DEVICE)
            original_vae = original_tensor
            original_spnn = original_tensor

        prev_vae_tensor = original_vae
        prev_spnn_tensor = original_spnn
        vae_all_tensors = [original_vae]
        spnn_all_tensors = [original_spnn]

        for c in range(num_cycles):
            prompt = args.prompts[c % len(args.prompts)]
            seed = 42 + c

            pipe_kwargs = dict(
                prompt=prompt,
                image=curr_vae_tensor,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                generator=torch.Generator(device=DEVICE).manual_seed(seed),
                output_type="pt",
            )
            if use_sd15:
                pipe_kwargs["strength"] = args.strength
            curr_vae_tensor = pipe_vae(**pipe_kwargs).images

            pipe_kwargs["image"] = curr_spnn_tensor
            pipe_kwargs["generator"] = torch.Generator(device=DEVICE).manual_seed(seed)
            curr_spnn_tensor = pipe_spnn(**pipe_kwargs).images

            vae_tensor_cur = curr_vae_tensor[0].cpu()
            spnn_tensor_cur = curr_spnn_tensor[0].cpu()

            vae_psnr_t = calc_psnr(vae_tensor_cur, original_vae)
            spnn_psnr_t = calc_psnr(spnn_tensor_cur, original_spnn)
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
              f"VAE total@10={vae_psnr_total_sum[-1]/n_done:.2f}dB  "
              f"SPNN total@10={spnn_psnr_total_sum[-1]/n_done:.2f}dB")

        # Save full cycle grids for the first few images
        # Top row: VAE (original + each cycle), Bottom row: SPNN (original + each cycle)
        if img_idx < args.num_save_grids:
            vae_row = torch.stack(vae_all_tensors)
            spnn_row = torch.stack(spnn_all_tensors)
            grid = torch.cat([vae_row, spnn_row], dim=0)
            grid_path = f"cycle_grid_img{img_idx:03d}.png"
            save_image(grid, grid_path,
                       nrow=num_cycles + 1, padding=2, pad_value=1.0)
            wandb.log({f"cycle_grid_img{img_idx}": wandb.Image(
                grid_path, caption=f"Top: VAE, Bottom: SPNN (img {img_idx})")})

    # Compute and log mean PSNR per cycle
    for c in range(num_cycles):
        n = num_images
        mean_vae_total = vae_psnr_total_sum[c] / n
        mean_spnn_total = spnn_psnr_total_sum[c] / n
        std_vae_total = np.sqrt(vae_psnr_total_sq[c] / n - mean_vae_total ** 2)
        std_spnn_total = np.sqrt(spnn_psnr_total_sq[c] / n - mean_spnn_total ** 2)
        mean_vae_step = vae_psnr_step_sum[c] / n
        mean_spnn_step = spnn_psnr_step_sum[c] / n

        print(f"Cycle {c+1}: VAE={mean_vae_total:.2f}±{std_vae_total:.2f}dB  "
              f"SPNN={mean_spnn_total:.2f}±{std_spnn_total:.2f}dB  "
              f"gap={mean_spnn_total - mean_vae_total:+.2f}dB")

        wandb.log({
            "cycle": c + 1,
            "mean_vae_psnr_vs_original": mean_vae_total,
            "mean_spnn_psnr_vs_original": mean_spnn_total,
            "std_vae_psnr_vs_original": std_vae_total,
            "std_spnn_psnr_vs_original": std_spnn_total,
            "mean_psnr_gap_vs_original": mean_spnn_total - mean_vae_total,
            "mean_vae_psnr_vs_prev": mean_vae_step,
            "mean_spnn_psnr_vs_prev": mean_spnn_step,
            "mean_psnr_gap_vs_prev": mean_spnn_step - mean_vae_step,
        })

    wandb.finish()
    print("Done.")


if __name__ == "__main__":
    main()
