"""
Latent DDNM (Denoising Diffusion Null-space Model) comparison: VAE vs SPNN.

At each reverse diffusion step the codec does decode → edit → encode.
SPNN's better cycle consistency should preserve more information across
the ~50 DDIM steps compared to the VAE.
"""

import argparse
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from torchvision.utils import save_image
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from dataset import CelebAHQDataset
from models import SPNNAutoencoder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ═══════════════════════════════════════════════════════════
# Degradation operators (pixel-space, linear)
# ═══════════════════════════════════════════════════════════

class SuperResolution4x:
    """A(x)=avg_pool(x,4), A_pinv(y)=nearest_upsample(y,4)."""

    name = "sr4x"

    def __init__(self, img_size):
        self.img_size = img_size

    def A(self, x):
        return F.avg_pool2d(x, 4)

    def A_pinv(self, y):
        return F.interpolate(y, scale_factor=4, mode="nearest")

    def null_space_project(self, x_0, y):
        return self.A_pinv(y) + x_0 - self.A_pinv(self.A(x_0))


class RandomInpainting:
    """50 % random pixel mask."""

    name = "inpaint"

    def __init__(self, img_size, mask_ratio=0.5, seed=42):
        gen = torch.Generator().manual_seed(seed)
        self.mask = (torch.rand(1, 1, img_size, img_size, generator=gen) > mask_ratio).float()

    def _mask(self, x):
        return self.mask.to(x.device)

    def A(self, x):
        return x * self._mask(x)

    def A_pinv(self, y):
        return y  # known pixels kept, zeros elsewhere

    def null_space_project(self, x_0, y):
        m = self._mask(x_0)
        return y * m + x_0 * (1 - m)


class GaussianDeblur:
    """Gaussian blur via FFT, Wiener deconvolution pseudo-inverse."""

    name = "deblur_gauss"

    def __init__(self, img_size, kernel_size=61, sigma=3.0, wiener_eps=0.01):
        self.img_size = img_size
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.wiener_eps = wiener_eps
        # Pre-compute blur kernel
        self.kernel = self._gaussian_kernel(kernel_size, sigma)
        # Pre-compute FFT of kernel (padded to image size)
        self._H_cache = {}

    @staticmethod
    def _gaussian_kernel(size, sigma):
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        kernel = torch.outer(g, g)
        return kernel / kernel.sum()

    def _get_H(self, x):
        key = (x.shape[-2], x.shape[-1], x.device)
        if key not in self._H_cache:
            h, w = x.shape[-2], x.shape[-1]
            pad_kernel = torch.zeros(h, w, device=x.device)
            kh, kw = self.kernel_size, self.kernel_size
            pad_kernel[:kh, :kw] = self.kernel.to(x.device)
            # Center the kernel
            pad_kernel = torch.roll(pad_kernel, (-kh // 2, -kw // 2), dims=(0, 1))
            H = torch.fft.fft2(pad_kernel)
            self._H_cache[key] = H
        return self._H_cache[key]

    def A(self, x):
        H = self._get_H(x)
        X = torch.fft.fft2(x)
        return torch.fft.ifft2(X * H).real

    def A_pinv(self, y):
        H = self._get_H(y)
        Y = torch.fft.fft2(y)
        # Wiener deconvolution
        H_conj = torch.conj(H)
        denom = H_conj * H + self.wiener_eps
        return torch.fft.ifft2(Y * H_conj / denom).real

    def null_space_project(self, x_0, y):
        return self.A_pinv(y) + x_0 - self.A_pinv(self.A(x_0))


DEGRADATION_REGISTRY = {
    "sr4x": SuperResolution4x,
    "inpaint": RandomInpainting,
    "deblur_gauss": GaussianDeblur,
}


# ═══════════════════════════════════════════════════════════
# Model loading
# ═══════════════════════════════════════════════════════════

def load_models(args):
    """Load SD UNet, scheduler, text encoder, VAE, and SPNN."""
    sd_id = "runwayml/stable-diffusion-v1-5"
    vae_id = "timbrooks/instruct-pix2pix"

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

    # Empty text embedding (inverse problem, not text-to-image)
    tokens = tokenizer("", padding="max_length", max_length=77,
                       return_tensors="pt").input_ids.to(DEVICE)
    with torch.no_grad():
        empty_emb = text_encoder(tokens).last_hidden_state  # [1, 77, 768]

    print("Loading VAE...")
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

    return unet, scheduler, vae, spnn, empty_emb


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
# DDNM reverse process
# ═══════════════════════════════════════════════════════════

@torch.no_grad()
def ddnm_sample(unet, scheduler, codec, degradation, y, empty_emb,
                guidance_scale=2.0):
    """
    Latent DDNM: reverse diffusion with null-space projection at each step.

    Args:
        codec: object with .encode(x) and .decode(z)
        degradation: object with .A(x), .A_pinv(y), .null_space_project(x0, y)
        y: degraded measurement in pixel space [B, 3, H, W]
        empty_emb: empty text embedding [1, 77, 768]
    Returns:
        Reconstructed image in pixel space [B, 3, H, W]
    """
    B = y.shape[0]
    latent_shape = (B, 4, y.shape[2] // 8, y.shape[3] // 8)
    emb = empty_emb.expand(B, -1, -1)

    # Start from pure noise
    z_t = torch.randn(latent_shape, device=DEVICE)

    alphas_cumprod = scheduler.alphas_cumprod.to(DEVICE)
    timesteps = scheduler.timesteps

    for i, t in enumerate(tqdm(timesteps, desc="DDNM steps", leave=False)):
        t_tensor = t.unsqueeze(0).expand(B).to(DEVICE)

        # 1. Predict noise (with CFG)
        if guidance_scale > 1.0:
            z_in = torch.cat([z_t, z_t], dim=0)
            t_in = t_tensor.repeat(2)
            e_in = torch.cat([torch.zeros_like(emb), emb], dim=0)
            noise_pred_all = unet(z_in, t_in, encoder_hidden_states=e_in).sample
            noise_uncond, noise_cond = noise_pred_all.chunk(2)
            noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
        else:
            noise_pred = unet(z_t, t_tensor, encoder_hidden_states=emb).sample

        # 2. Estimate clean latent z_0
        alpha_t = alphas_cumprod[t].view(1, 1, 1, 1)
        z_0_pred = (z_t - (1 - alpha_t).sqrt() * noise_pred) / alpha_t.sqrt()

        # 3. Decode to pixel space
        x_0_pred = codec.decode(z_0_pred)

        # 4. Null-space projection (data consistency)
        x_0_corrected = degradation.null_space_project(x_0_pred, y)

        # 5. Re-encode to latent space
        z_0_corrected = codec.encode(x_0_corrected)

        # 6. DDIM step to z_{t-1}
        if i < len(timesteps) - 1:
            t_next = timesteps[i + 1]
            alpha_next = alphas_cumprod[t_next].view(1, 1, 1, 1)
        else:
            alpha_next = torch.tensor(1.0, device=DEVICE).view(1, 1, 1, 1)

        z_t = alpha_next.sqrt() * z_0_corrected + (1 - alpha_next).sqrt() * noise_pred

    # Final decode
    return codec.decode(z_t)


# ═══════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════

def psnr(pred, target):
    """PSNR in dB, images in [-1, 1]."""
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return float("inf")
    # Range is 2.0 for [-1, 1]
    return 10 * math.log10(4.0 / mse.item())


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

def run(args):
    unet, scheduler, vae, spnn, empty_emb = load_models(args)

    codecs = {
        "VAE": VAECodec(vae),
        "SPNN": SPNNCodec(spnn),
    }

    # Load test images
    dataset = CelebAHQDataset(img_size=args.img_size, split="test", n_test=1000)
    images = torch.stack([dataset[i] for i in range(args.num_images)]).to(DEVICE)
    print(f"Loaded {images.shape[0]} test images ({images.shape[1:]})")

    for problem_name in args.problems:
        print(f"\n{'=' * 60}")
        print(f"Problem: {problem_name}")
        print(f"{'=' * 60}")

        # Create degradation operator
        deg_cls = DEGRADATION_REGISTRY[problem_name]
        degradation = deg_cls(args.img_size)

        out_dir = os.path.join("ddnm_results", problem_name)
        os.makedirs(out_dir, exist_ok=True)

        all_metrics = {name: {"psnr": [], "mse": []} for name in codecs}

        for img_idx in range(images.shape[0]):
            x_gt = images[img_idx : img_idx + 1]  # [1, 3, H, W]

            # Create degraded measurement
            y = degradation.A(x_gt)

            results = {}
            for codec_name, codec in codecs.items():
                print(f"  Image {img_idx + 1}/{images.shape[0]}, codec={codec_name}...")
                x_recon = ddnm_sample(
                    unet, scheduler, codec, degradation, y, empty_emb,
                    guidance_scale=args.guidance_scale,
                )
                # Clamp to valid range
                x_recon = x_recon.clamp(-1, 1)
                results[codec_name] = x_recon

                # Metrics
                p = psnr(x_recon, x_gt)
                m = F.mse_loss(x_recon, x_gt).item()
                all_metrics[codec_name]["psnr"].append(p)
                all_metrics[codec_name]["mse"].append(m)
                print(f"    PSNR: {p:.2f} dB, MSE: {m:.6f}")

            # Save grid: Degraded | VAE result | SPNN result | Ground truth
            # For SR, upsample degraded for display
            if problem_name == "sr4x":
                y_display = F.interpolate(y, scale_factor=4, mode="nearest")
            else:
                y_display = y

            grid_images = [
                (y_display[0].cpu() + 1) / 2,
                (results["VAE"][0].cpu() + 1) / 2,
                (results["SPNN"][0].cpu() + 1) / 2,
                (x_gt[0].cpu() + 1) / 2,
            ]
            grid = torch.stack(grid_images)
            save_image(grid, os.path.join(out_dir, f"img{img_idx:03d}.png"),
                       nrow=4, padding=2)

        # Print summary
        print(f"\n--- {problem_name} Summary ---")
        for codec_name in codecs:
            avg_psnr = sum(all_metrics[codec_name]["psnr"]) / len(all_metrics[codec_name]["psnr"])
            avg_mse = sum(all_metrics[codec_name]["mse"]) / len(all_metrics[codec_name]["mse"])
            print(f"  {codec_name:>5s}: avg PSNR = {avg_psnr:.2f} dB, avg MSE = {avg_mse:.6f}")

    print("\nDone. Results saved to ddnm_results/")


def parse_args():
    parser = argparse.ArgumentParser(description="Latent DDNM: VAE vs SPNN comparison")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="SPNN model checkpoint path")
    parser.add_argument("--problems", nargs="+", default=["sr4x", "inpaint"],
                        choices=list(DEGRADATION_REGISTRY.keys()),
                        help="Inverse problems to run")
    parser.add_argument("--num_images", type=int, default=5,
                        help="Number of test images")
    parser.add_argument("--num_steps", type=int, default=50,
                        help="DDIM steps")
    parser.add_argument("--guidance_scale", type=float, default=2.0,
                        help="Classifier-free guidance scale")
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
