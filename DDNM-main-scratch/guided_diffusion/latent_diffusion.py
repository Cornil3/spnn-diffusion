"""
Latent-space DDNM runner -- a near-line-for-line latent port of
guided_diffusion/diffusion.py (the simplified, no-SVD DDNM+ path).

Read this file side-by-side with diffusion.py: same structure (get_beta_schedule,
the Diffusion class, the simplified_ddnm_plus loop, the bottom helpers). The loop
is changed in exactly FOUR places, all latent:
  (1) the diffused variable is the SD latent z, not the pixel image x;
  (2) model(xt, t)                  -> unet(zt, t, context)  (SD1.5 UNet is
      conditional, so context = the empty-prompt "null" embedding -> unconditional);
  (3) the back-projection (bp) line is wrapped by the codec, the only place pixels
      appear:   x0 = decode(z0);  x0_hat = x0 - lambda*Ap(A x0 - y);  z0 = encode(x0_hat);
  (4) the result is decode(z).
Everything else (Eq. 19 lambda_t/gamma_t, the DDIM step, RePaint time-travel) is
identical to the original.
"""

import os
import glob
import logging

import numpy as np
import torch
import tqdm
import torch.utils.data as data
import torchvision.utils as tvu
from torchvision import transforms
from PIL import Image

from functions.latent_codec import load_vae, load_spnn, SD15_ID


# --------------------------------------------------------------------------- #
# Pixel-space degradation helpers (verbatim from diffusion.py)
# --------------------------------------------------------------------------- #
def get_gaussian_noisy_img(img, noise_level):
    return img + torch.randn_like(img) * noise_level


def MeanUpsample(x, scale):
    n, c, h, w = x.shape
    out = torch.zeros(n, c, h, scale, w, scale).to(x.device) + x.view(n, c, h, 1, w, 1)
    out = out.view(n, c, scale * h, scale * w)
    return out


def color2gray(x):
    coef = 1 / 3
    x = x[:, 0, :, :] * coef + x[:, 1, :, :] * coef + x[:, 2, :, :] * coef
    return x.repeat(1, 3, 1, 1)


def gray2color(x):
    x = x[:, 0, :, :]
    coef = 1 / 3
    base = coef ** 2 + coef ** 2 + coef ** 2
    return torch.stack((x * coef / base, x * coef / base, x * coef / base), 1)


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    if beta_schedule == "scaled_linear":          # SD1.x / 2.x
        betas = (
            np.linspace(beta_start ** 0.5, beta_end ** 0.5,
                        num_diffusion_timesteps, dtype=np.float64) ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


def inverse_data_transform(x):
    return ((x + 1.0) / 2.0).clamp(0.0, 1.0)


class LatentDiffusion(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_schedule=getattr(config.diffusion, "beta_schedule", "scaled_linear"),
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        )
        self.alphas_cumprod_prev = alphas_cumprod_prev
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

        self.sd_id = getattr(config.model, "sd15_id", SD15_ID)
        self.latent_channels = 4
        self.f = 8  # SD VAE downsample factor

    def sample(self, simplified=True):
        # --- load the diffusion prior (SD1.5 UNet) + the unconditional context ---
        # Replaces diffusion.py's pixel-model loading. SD1.5's UNet is conditional,
        # so "unconditional" means feeding the empty-prompt ("null") CLIP embedding.
        from diffusers import UNet2DConditionModel
        from transformers import CLIPTextModel, CLIPTokenizer

        logging.info("Loading SD1.5 UNet + text encoder (%s)", self.sd_id)
        unet = UNet2DConditionModel.from_pretrained(self.sd_id, subfolder="unet")
        unet.eval().to(self.device)
        unet.requires_grad_(False)

        tok = CLIPTokenizer.from_pretrained(self.sd_id, subfolder="tokenizer")
        te = CLIPTextModel.from_pretrained(self.sd_id, subfolder="text_encoder")
        te.eval().to(self.device)
        with torch.no_grad():
            prompt = getattr(self.args, "prompt", "") or ""        # "" => unconditional
            ids = tok([prompt], padding="max_length", max_length=tok.model_max_length,
                      return_tensors="pt").input_ids.to(self.device)
            context = te(ids)[0]

        # --- codecs: VAE, plus SPNN-512 if requested (replaces the single pixel model) ---
        codecs = [("VAE", load_vae(self.device))]
        if getattr(self.config.codec, "type", "vae") == "spnn":
            codecs.append(("SPNN", load_spnn(self.config, self.device)))

        print('Run latent DDNM (simplified, no SVD).',
              f'{self.config.time_travel.T_sampling} sampling steps.',
              f'travel_length = {self.config.time_travel.travel_length},',
              f'travel_repeat = {self.config.time_travel.travel_repeat}.',
              f'Task: {self.args.deg}.',
              f'Codecs: {[c[0] for c in codecs]}.')
        self.simplified_latent_ddnm_plus(unet, context, codecs)

    def simplified_latent_ddnm_plus(self, unet, context, codecs):
        args, config = self.args, self.config
        device = self.device
        size = int(config.data.image_size)

        val_loader = self._dataloader()
        print(f'Dataset has size {len(val_loader.dataset)}')

        # get degradation operator (pixel space) -- same block as diffusion.py
        print("args.deg:", args.deg)
        if args.deg == 'colorization':
            A = lambda z: color2gray(z)
            Ap = lambda z: gray2color(z)
        elif args.deg == 'denoising':
            A = lambda z: z
            Ap = A
        elif args.deg == 'sr_averagepooling':
            scale = round(args.deg_scale)
            A = torch.nn.AdaptiveAvgPool2d((size // scale, size // scale))
            Ap = lambda z: MeanUpsample(z, scale)
        elif args.deg == 'inpainting':
            mask = self._load_mask(size)            # [1,1,H,W], 1 = known, 0 = hole
            A = lambda z: z * mask
            Ap = A
        else:
            raise NotImplementedError("degradation type not supported")

        args.sigma_y = 2 * args.sigma_y             # account for [-1,1] scaling
        sigma_y = args.sigma_y

        # reverse-diffusion schedule (shared across images and codecs)
        skip = config.diffusion.num_diffusion_timesteps // config.time_travel.T_sampling
        times = get_schedule_jump(config.time_travel.T_sampling,
                                  config.time_travel.travel_length,
                                  config.time_travel.travel_repeat)
        time_pairs = list(zip(times[:-1], times[1:]))
        total_steps = sum(1 for a, b in time_pairs
                          if (b * skip if b >= 0 else -1) < a * skip)

        idx_so_far = 0
        avg_psnr = {name: 0.0 for name, _ in codecs}
        pbar = tqdm.tqdm(val_loader)
        for x_orig, classes in pbar:
            x_orig = x_orig.to(device) * 2.0 - 1.0   # data_transform: [0,1] -> [-1,1]
            y = A(x_orig)

            if config.sampling.batch_size != 1:
                raise ValueError("please change the config file to set batch size as 1")

            os.makedirs(os.path.join(args.image_folder, "Apy"), exist_ok=True)
            tvu.save_image(inverse_data_transform(Ap(y)[0]),
                           os.path.join(args.image_folder, f"Apy/Apy_{idx_so_far}.png"))
            tvu.save_image(inverse_data_transform(x_orig[0]),
                           os.path.join(args.image_folder, f"Apy/orig_{idx_so_far}.png"))

            # run each codec under the SAME UNet / schedule (the SPNN-vs-VAE comparison)
            for codec_name, codec in codecs:
                os.makedirs(os.path.join(args.image_folder, codec_name), exist_ok=True)

                # init z_T  (latent Gaussian; replaces the pixel x_T)
                z = torch.randn(1, self.latent_channels, size // self.f, size // self.f,
                                device=device)

                with torch.no_grad():
                    n = z.size(0)
                    z0_preds = []
                    zs = [z]
                    step_count = 0

                    # reverse diffusion sampling
                    for i, j in time_pairs:
                        i, j = i * skip, j * skip
                        if j < 0: j = -1

                        if j < i:  # normal sampling
                            step_count += 1
                            t = (torch.ones(n) * i).to(device)
                            next_t = (torch.ones(n) * j).to(device)
                            at = compute_alpha(self.betas, t.long())
                            at_next = compute_alpha(self.betas, next_t.long())
                            sigma_t = (1 - at_next ** 2).sqrt()
                            zt = zs[-1].to(device)

                            # (2) latent UNet replaces model(xt, t); context = null embedding
                            et = unet(zt, t.long(), encoder_hidden_states=context).sample

                            # Eq. 12
                            z0_t = (zt - et * (1 - at).sqrt()) / at.sqrt()

                            # Eq. 19 residual-noise scale gamma_t (lambda_t comes from the schedule)
                            if sigma_t >= at_next * sigma_y:
                                gamma_t = (sigma_t ** 2 - (at_next * sigma_y) ** 2).sqrt()
                            else:
                                gamma_t = 0.
                            # data-consistency strength lambda_t (const=1.0 default; or decay schedule)
                            lambda_t = self._lambda_schedule(step_count, total_steps, at_next)

                            # (3) Eq. 17 back-projection in PIXEL space: decode -> bp -> encode
                            x0_t = codec.decode(z0_t)
                            x0_t_hat = x0_t - lambda_t * Ap(A(x0_t) - y)
                            z0_t_hat = codec.encode(x0_t_hat)

                            eta = self.args.eta
                            c1 = (1 - at_next).sqrt() * eta
                            c2 = (1 - at_next).sqrt() * ((1 - eta ** 2) ** 0.5)
                            # damp the et (direction) term toward the end of sampling (damping_floor)
                            damping_floor = float(getattr(self.args, "damping_floor", 1.0))
                            damp = 1.0 - at_next.item() * (1.0 - damping_floor)
                            # different from the paper, we use DDIM here instead of DDPM
                            zt_next = at_next.sqrt() * z0_t_hat + gamma_t * (c1 * torch.randn_like(z0_t) + damp * c2 * et)

                            z0_preds.append(z0_t.to('cpu'))
                            zs.append(zt_next.to('cpu'))
                        else:  # time-travel back
                            next_t = (torch.ones(n) * j).to(device)
                            at_next = compute_alpha(self.betas, next_t.long())
                            z0_t = z0_preds[-1].to(device)
                            zt_next = at_next.sqrt() * z0_t + torch.randn_like(z0_t) * (1 - at_next).sqrt()
                            zs.append(zt_next.to('cpu'))

                    z = zs[-1].to(device)

                # (4) decode the final latent back to pixels
                x0 = codec.decode(z)
                if args.deg == 'inpainting' and bool(getattr(args, 'final_known_pin', True)):
                    # final_known_pin: force the unmasked region EXACTLY to the measurement.
                    # Sampling is unchanged; this only rewrites the known region of the output.
                    # mask: 1=known, 0=hole; y = A(x_orig) = mask*x_orig (hole stays the sample).
                    x0 = mask * y + (1.0 - mask) * x0
                x = inverse_data_transform(x0)

                tvu.save_image(x[0], os.path.join(args.image_folder, codec_name,
                                                  f"{idx_so_far}_recon.png"))
                orig = inverse_data_transform(x_orig[0])
                mse = torch.mean((x[0] - orig) ** 2)
                psnr = 10 * torch.log10(1 / mse)
                avg_psnr[codec_name] += psnr.item()

            idx_so_far += 1
            pbar.set_description(
                " | ".join(f"{n} {avg_psnr[n] / idx_so_far:.2f}dB" for n, _ in codecs)
            )

        print(f"\nTotal Average PSNR over {idx_so_far} images:")
        for name, _ in codecs:
            print(f"  {name:5s}: {avg_psnr[name] / max(1, idx_so_far):.2f} dB")
        with open(os.path.join(args.image_folder, "PSNR.txt"), "w") as fpsnr:
            for name, _ in codecs:
                fpsnr.write(f"{name} {avg_psnr[name] / max(1, idx_so_far):.2f} dB\n")

    # ----------------------------------------------------------------- helpers
    def _lambda_schedule(self, step_count, total_steps, at_next):
        """Data-consistency strength lambda_t over sampling (like DDNM-main/diffusion.py).

        const : lambda_val (flat).
        exp   : decay lambda_val -> lambda_floor, exp tempo lambda_rate (>0).
        linear: decay lambda_val -> lambda_floor, linearly.
        noise : lambda_floor + (lambda_val-lambda_floor)*(1 - a_next)  (strong early).
        """
        import math
        a = self.args
        mode = getattr(a, "lambda_mode", "const")
        lam_val = float(getattr(a, "lambda_val", 1.0))      # peak / step-0
        lam_floor = float(getattr(a, "lambda_floor", 0.0))  # asymptotic floor
        if mode == "const":
            return lam_val
        if mode == "noise":
            return lam_floor + (lam_val - lam_floor) * float(1.0 - at_next.item())
        progress = min(1.0, (step_count - 1) / max(1, total_steps - 1))  # 0 -> 1
        if mode == "exp":
            rate = float(getattr(a, "lambda_rate", 5.0))
            if rate > 0.0:
                e1 = math.exp(-rate)
                curve = (math.exp(-rate * progress) - e1) / (1.0 - e1)
            else:
                curve = 1.0 - progress
        else:  # linear
            curve = 1.0 - progress
        return lam_floor + (lam_val - lam_floor) * curve

    def _dataloader(self):
        size = int(self.config.data.image_size)
        root = self.config.data.dataset_root
        if not os.path.isabs(root):
            root = os.path.abspath(root)
        tf = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(size),
            transforms.ToTensor(),   # -> [0, 1]
        ])
        paths = sorted(p for p in glob.glob(os.path.join(root, "**", "*"), recursive=True)
                       if p.lower().endswith((".png", ".jpg", ".jpeg"))
                       and ".ipynb_checkpoints" not in p)
        if not paths:
            raise FileNotFoundError(f"no images under {root!r}")
        start = int(getattr(self.args, "subset_start", -1))
        end = int(getattr(self.args, "subset_end", -1))
        if start >= 0 and end > 0:
            paths = paths[start:end]

        class _DS(data.Dataset):
            def __len__(s): return len(paths)
            def __getitem__(s, k): return tf(Image.open(paths[k]).convert("RGB")), 0

        return data.DataLoader(_DS(), batch_size=1, shuffle=False,
                               num_workers=int(getattr(self.config.data, "num_workers", 0)))

    def _load_mask(self, size):
        """Inpainting mask [1,1,H,W], 1 = known, 0 = hole. Uses --mask, else
        config.inpainting.mask_path, else a centered box."""
        inp = getattr(self.config, "inpainting", None)
        path = getattr(self.args, "mask", "") or (getattr(inp, "mask_path", "") if inp else "")
        if path and os.path.exists(path):
            if path.endswith(".npy"):
                m = torch.from_numpy(np.load(path)).float()
            else:
                m = (torch.from_numpy(np.array(Image.open(path).convert("L"))).float()
                     / 255.0 > 0.5).float()
            m = m.view(1, 1, *m.shape[-2:])
            if m.shape[-1] != size or m.shape[-2] != size:
                m = torch.nn.functional.interpolate(m, size=(size, size), mode="nearest")
            return m.to(self.device)
        # fallback: centered box hole
        m = torch.ones(1, 1, size, size, device=self.device)
        s = size // 2
        m[..., size // 4:size // 4 + s, size // 4:size // 4 + s] = 0.0
        return m


# Code from RePaint (verbatim from diffusion.py / svd_ddnm.py)
def get_schedule_jump(T_sampling, travel_length, travel_repeat):
    jumps = {}
    for j in range(0, T_sampling - travel_length, travel_length):
        jumps[j] = travel_repeat - 1

    t = T_sampling
    ts = []
    while t >= 1:
        t = t - 1
        ts.append(t)
        if jumps.get(t, 0) > 0:
            jumps[t] = jumps[t] - 1
            for _ in range(travel_length):
                t = t + 1
                ts.append(t)
    ts.append(-1)
    _check_times(ts, -1, T_sampling)
    return ts


def _check_times(times, t_0, T_sampling):
    assert times[0] > times[1], (times[0], times[1])
    assert times[-1] == -1, times[-1]
    for t_last, t_cur in zip(times[:-1], times[1:]):
        assert abs(t_last - t_cur) == 1, (t_last, t_cur)
    for t in times:
        assert t >= t_0, (t, t_0)
        assert t <= T_sampling, (t, T_sampling)


def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a
