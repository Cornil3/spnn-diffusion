"""
Latent-space DDNM runner.

Loads the SD1.5 UNet (diffusion prior), its noise schedule and text encoder, plus
one or both codecs (KL-VAE and SPNN-512), then runs latent DDNM inpainting over an
ImageFolder of test images and reports PSNR per codec.

Dispatched from main.py when `config.model.type` starts with "latent". The
official pixel-space `Diffusion` runner is left untouched.
"""

import os
import glob
import logging

import numpy as np
import torch
import torch.utils.data as data
import torchvision.utils as tvu
from torchvision import transforms
from PIL import Image

from functions.latent_codec import load_vae, load_spnn, SD15_ID
from functions.latent_ddnm import (
    sd15_alphas_cumprod, make_alpha_bar, make_inpaint_mask, load_mask_file,
    latent_ddnm_sample,
)

_IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".JPEG", ".JPG", ".PNG")


def _to01(x):
    """[-1, 1] -> [0, 1] for saving / PSNR."""
    return ((x + 1.0) / 2.0).clamp(0.0, 1.0)


class _PlainImageDataset(data.Dataset):
    """All images found recursively under `root`, skipping hidden/checkpoint dirs.

    More forgiving than torchvision.ImageFolder (which requires non-empty class
    subfolders and would choke on a stray `.ipynb_checkpoints/`).
    """

    def __init__(self, root, transform):
        self.transform = transform
        paths = []
        for p in glob.glob(os.path.join(root, "**", "*"), recursive=True):
            if not p.lower().endswith(tuple(e.lower() for e in _IMG_EXTS)):
                continue
            if any(part.startswith(".") for part in os.path.relpath(p, root).split(os.sep)):
                continue
            paths.append(p)
        self.paths = sorted(paths)
        if not self.paths:
            raise FileNotFoundError(f"no images found under {root!r}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.transform(img), 0


class LatentDiffusion(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.sd_id = getattr(config.model, "sd15_id", SD15_ID)

    # ------------------------------------------------------------------ models
    def _load_prior(self):
        from diffusers import UNet2DConditionModel
        from transformers import CLIPTextModel, CLIPTokenizer

        logging.info("Loading SD1.5 UNet + text encoder (%s)", self.sd_id)
        unet = UNet2DConditionModel.from_pretrained(self.sd_id, subfolder="unet")
        unet.eval().to(self.device)
        for p in unet.parameters():
            p.requires_grad_(False)

        # SD1.5's exact training schedule (scaled_linear), built analytically so we
        # depend only on the cached UNet/VAE/text-encoder, never the scheduler config.
        d = getattr(self.config, "diffusion", None)
        alphas_cumprod = sd15_alphas_cumprod(
            num_train_timesteps=int(getattr(d, "num_diffusion_timesteps", 1000)) if d else 1000,
            beta_start=float(getattr(d, "beta_start", 0.00085)) if d else 0.00085,
            beta_end=float(getattr(d, "beta_end", 0.012)) if d else 0.012,
            device=self.device,
        )

        tokenizer = CLIPTokenizer.from_pretrained(self.sd_id, subfolder="tokenizer")
        text_encoder = CLIPTextModel.from_pretrained(self.sd_id, subfolder="text_encoder")
        text_encoder.eval().to(self.device)
        for p in text_encoder.parameters():
            p.requires_grad_(False)
        return unet, alphas_cumprod, tokenizer, text_encoder

    @torch.no_grad()
    def _embed(self, tokenizer, text_encoder, prompt):
        tok = tokenizer(
            prompt, padding="max_length", max_length=tokenizer.model_max_length,
            truncation=True, return_tensors="pt",
        )
        return text_encoder(tok.input_ids.to(self.device))[0]

    def _build_codecs(self):
        """Return [(name, codec), ...]. 'spnn' runs VAE then SPNN for a head-to-head."""
        codec_type = getattr(self.config.codec, "type", "vae")
        codecs = [("VAE", load_vae(self.device))]
        if codec_type == "spnn":
            codecs.append(("SPNN", load_spnn(self.config, self.device)))
        return codecs

    # -------------------------------------------------------------------- data
    def _dataloader(self):
        size = int(self.config.data.image_size)
        root = self.config.data.dataset_root
        if not os.path.isabs(root):
            root = os.path.abspath(root)
        tf = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(size),
            transforms.ToTensor(),  # -> [0, 1]
        ])
        ds = _PlainImageDataset(root, transform=tf)
        # Honor the official --subset_start/--subset_end (handy for cheap smoke tests).
        start = int(getattr(self.args, "subset_start", -1))
        end = int(getattr(self.args, "subset_end", -1))
        if start >= 0 and end > 0:
            ds = data.Subset(ds, range(start, min(end, len(ds))))
        loader = data.DataLoader(
            ds, batch_size=1, shuffle=False,
            num_workers=int(getattr(self.config.data, "num_workers", 0)),
        )
        return loader

    def _build_mask(self, size):
        """Return (mask[1,1,H,W] with 1=known, kind_str, frac) for logging."""
        args, config = self.args, self.config
        inp = getattr(config, "inpainting", None)
        if getattr(args, "mask", ""):
            return load_mask_file(args.mask, size, self.device), os.path.basename(args.mask), 0.0
        kind = getattr(args, "mask_kind", "") or (getattr(inp, "mask_kind", "box") if inp else "box")
        frac = getattr(args, "mask_box_frac", -1.0)
        if frac is None or frac < 0:
            frac = float(getattr(inp, "mask_box_frac", 0.5)) if inp else 0.5
        return make_inpaint_mask(size, kind=kind, box_frac=frac, device=self.device), kind, frac

    # -------------------------------------------------------------------- main
    def sample(self):
        args, config = self.args, self.config
        if args.deg != "inpainting":
            raise NotImplementedError(
                f"latent DDNM currently implements inpainting; got deg={args.deg!r}"
            )

        unet, alphas_cumprod, tokenizer, text_encoder = self._load_prior()
        alpha_bar = make_alpha_bar(alphas_cumprod)
        ntt = alphas_cumprod.numel()
        codecs = self._build_codecs()

        # text context (empty prompt => SD's unconditional embedding)
        prompt = getattr(args, "prompt", "") or ""
        guidance = float(getattr(args, "guidance_scale", -1.0))
        if guidance < 0:
            guidance = float(getattr(config.model, "guidance_scale", 1.0))
        context = self._embed(tokenizer, text_encoder, prompt)
        context_uncond = self._embed(tokenizer, text_encoder, "") if guidance != 1.0 else None

        size = int(config.data.image_size)
        mask, mask_kind, mask_frac = self._build_mask(size)

        # Inpainting operator in pixel space. A^+ of a 0/1 projection is itself.
        A = lambda x: mask * x
        Ap = lambda r: mask * r

        T = int(config.time_travel.T_sampling)
        eta = float(getattr(args, "eta", 0.85))
        latent_shape = (1, 4, size // 8, size // 8)

        g = torch.Generator(device=self.device).manual_seed(int(getattr(args, "seed", 1234)))

        loader = self._dataloader()
        n_imgs = len(loader.dataset)
        logging.info(
            "Latent DDNM | imgs=%d codecs=%s T=%d eta=%.2f guidance=%.2f prompt=%r mask=%s(%.2f)",
            n_imgs, [c[0] for c in codecs], T, eta, guidance, prompt, mask_kind, mask_frac,
        )

        psnr_acc = {name: 0.0 for name, _ in codecs}
        for idx, (x_orig, _) in enumerate(loader):
            x_orig = x_orig.to(self.device) * 2.0 - 1.0   # [-1, 1]
            y = A(x_orig)                                  # masked measurement (known pixels)

            for name, codec in codecs:
                out_dir = os.path.join(args.image_folder, name)
                os.makedirs(out_dir, exist_ok=True)

                x_rec = latent_ddnm_sample(
                    unet, codec, alpha_bar, context, A, Ap, y,
                    T_sampling=T, num_train_timesteps=ntt, eta=eta,
                    guidance_scale=guidance, context_uncond=context_uncond,
                    latent_shape=latent_shape, device=self.device, generator=g,
                )

                rec01, orig01 = _to01(x_rec), _to01(x_orig)
                mse = torch.mean((rec01 - orig01) ** 2)
                psnr = (10 * torch.log10(1.0 / mse)).item()
                psnr_acc[name] += psnr

                tvu.save_image(rec01, os.path.join(out_dir, f"{idx:04d}_recon.png"))
                tvu.save_image(_to01(y), os.path.join(out_dir, f"{idx:04d}_masked.png"))
                tvu.save_image(orig01, os.path.join(out_dir, f"{idx:04d}_orig.png"))
                logging.info("[%-4s] img %d/%d  PSNR = %.2f dB", name, idx + 1, n_imgs, psnr)

        print(f"\n==== Latent DDNM | avg PSNR over {n_imgs} imgs "
              f"(T={T}, eta={eta}, mask={mask_kind}/{mask_frac}) ====")
        for name, _ in codecs:
            print(f"  {name:5s}: {psnr_acc[name] / max(1, n_imgs):.2f} dB")
