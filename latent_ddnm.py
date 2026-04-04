"""
Latent-space DDNM with SD UNet:
  - **in_channels=4** (base SD v1.5): UNet sees noisy latents only.
  - **in_channels=9** (inpainting): UNet sees [latents | mask | masked-image latents].

Pixel null-space projection + re-encode each step; schedule from simplified_ddnm_plus.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from tqdm import tqdm

from functions.svd_ddnm import get_schedule_jump


def latent_ddnm_inpaint_simplified(
    z_init: torch.Tensor,
    y_m11: torch.Tensor,
    mask: torch.Tensor,
    unet,
    vae,
    scheduler,
    prompt_embeds: torch.Tensor,
    alphas_cumprod: torch.Tensor,
    *,
    num_train_timesteps: int,
    T_sampling: int,
    travel_length: int,
    travel_repeat: int,
    eta: float,
    sigma_y: float,
    device: torch.device,
    return_snapshots: bool = False,
    mask_latent_interpolation: str = "bilinear",
):
    """
    z_init: (B, 4, h, w) latent noise
    y_m11: A(x) = x * mask in [-1,1] (known pixels only); mask 1 = known, 0 = hole (DDNM convention)
    mask: (B, 1, H, W) in {0,1}, 1 = keep / observe
    unet: SD UNet with **in_channels=4** (base v1.5) or **9** (inpainting checkpoint).
    return_snapshots: if True, return (z_final, list of x0_hat in [0,1] after each main reverse step).
    mask_latent_interpolation: ``\"nearest\"`` or ``\"bilinear\"`` for downsampling the inpainting mask
        to latent resolution. ``nearest`` matches a hard stair-step mask (often looks **boxy** at the
        hole boundary); ``bilinear`` softens the latent mask slightly and usually reduces blocky edges.
    """
    ic = unet.config.in_channels
    if ic not in (4, 9):
        raise ValueError(
            f"latent_ddnm_inpaint_simplified expects UNet in_channels 4 or 9, got {ic}."
        )
    use_inpaint_unet = ic == 9

    # Align first-stage VAE with `device` (e.g. SPNN loaded on CPU while UNet/z use CUDA).
    if hasattr(vae, "to"):
        vae.to(device)

    skip = num_train_timesteps // T_sampling
    n = z_init.size(0)
    z0_preds = []
    zs = [z_init]

    if mask.shape[1] == 1:
        mask3 = mask.expand(-1, 3, -1, -1)
    else:
        mask3 = mask

    times = get_schedule_jump(T_sampling, travel_length, travel_repeat)
    time_pairs = list(zip(times[:-1], times[1:]))

    sigma_y = 2.0 * sigma_y

    try:
        scaling = float(getattr(vae.config, "scaling_factor", 1.0))
    except:
        scaling = 1.0
    _, _, h_lat, w_lat = z_init.shape

    def decode_m11(z):
        z_in = z / scaling
        out = vae.decode(z_in)
        return out.sample if hasattr(out, "sample") else out

    def encode_m11(x_m11):
        enc = vae.encode(x_m11)
        if hasattr(enc, "latent_dist") and hasattr(enc.latent_dist, "mode"):
            z = enc.latent_dist.mode()
        else:
            z = enc
        return z * scaling

    # --- Inpainting UNet only: latent mask + masked-image latents (diffusers layout) ---
    if use_inpaint_unet:
        mask_sd = 1.0 - mask.to(device)
        mode = (mask_latent_interpolation or "bilinear").lower()
        if mode not in ("nearest", "bilinear"):
            mode = "bilinear"
        interp_kw: dict = {"size": (h_lat, w_lat), "mode": mode}
        if mode == "bilinear":
            interp_kw["align_corners"] = False
        mask_latent = F.interpolate(mask_sd, **interp_kw).to(device=device, dtype=z_init.dtype)
        masked_image_latents = encode_m11(y_m11.to(device)).to(device=device, dtype=z_init.dtype)
    else:
        mask_latent = None
        masked_image_latents = None

    def alpha_bar_next(j_step: int, next_t_tensor: torch.Tensor) -> torch.Tensor:
        if j_step == -1:
            return torch.ones(n, 1, 1, 1, device=device, dtype=alphas_cumprod.dtype)
        return alphas_cumprod[next_t_tensor].view(-1, 1, 1, 1)

    step_snapshots: list = [] if return_snapshots else None

    with torch.no_grad():
        desc = "latent DDNM (inpaint-UNet)" if use_inpaint_unet else "latent DDNM (SD v1.5 UNet)"
        for i, j in tqdm(time_pairs, desc=desc):
            i, j = i * skip, j * skip
            if j < 0:
                j = -1

            if j < i:
                t = torch.full((n,), i, device=device, dtype=torch.long)
                next_t = torch.full((n,), j, device=device, dtype=torch.long)

                at = alphas_cumprod[t].view(-1, 1, 1, 1)
                at_next = alpha_bar_next(j, next_t)

                sigma_t = (1.0 - at_next**2).sqrt()

                zt = zs[-1].to(device)

                # (1) Latent UNet input: 4 ch (base) or 9 ch [z_t | mask | masked_lat] (inpainting)
                t0 = int(t[0].item())
                z_scaled = scheduler.scale_model_input(zt, t0)
                if use_inpaint_unet:
                    latent_model_input = torch.cat([z_scaled, mask_latent, masked_image_latents], dim=1)
                else:
                    latent_model_input = z_scaled

                noise_pred = unet(latent_model_input, t, encoder_hidden_states=prompt_embeds).sample

                # Eq. 12 — latent x0 from ε prediction (used below)
                z0_t = (zt - noise_pred * (1.0 - at).sqrt()) / at.sqrt()

                # (3) Decode — z_{0|t} → pixels for measurement / projection
                x0_t = decode_m11(z0_t)

                cond = sigma_t >= at_next * sigma_y
                lambda_t = torch.where(cond, torch.ones_like(at), sigma_t / (at_next * sigma_y + 1e-20))
                gamma_t = torch.where(
                    cond,
                    (sigma_t**2 - (at_next * sigma_y) ** 2).clamp(min=0).sqrt(),
                    torch.zeros_like(sigma_t),
                )

                # (4) Back-projection — simplified DDNM inpainting in pixel space
                x0_t_hat = x0_t - lambda_t * mask3 * (mask3 * x0_t - y_m11)

                if return_snapshots:
                    step_snapshots.append(((x0_t_hat.clamp(-1, 1) + 1.0) / 2.0).detach().cpu())

                # (5) Encode — x0_hat → z0_hat; then ε̂ and z_{t-1}
                z0_hat = encode_m11(x0_t_hat)

                denom = (1.0 - at).sqrt().clamp(min=1e-6)
                eps_hat = (zt - at.sqrt() * z0_hat) / denom

                c1 = (1.0 - at_next).sqrt() * eta
                c2 = (1.0 - at_next).sqrt() * ((1.0 - eta**2) ** 0.5)

                z_next = at_next.sqrt() * z0_hat + gamma_t * (
                    c1 * torch.randn_like(z0_hat) + c2 * eps_hat
                )

                z0_preds.append(z0_t.cpu())
                zs.append(z_next.cpu())
            else:
                next_t = torch.full((n,), j, device=device, dtype=torch.long)
                at_next = alpha_bar_next(j, next_t)
                z0_t = z0_preds[-1].to(device)

                z_next = at_next.sqrt() * z0_t + torch.randn_like(z0_t) * (1.0 - at_next).sqrt()

                zs.append(z_next.cpu())

    final = zs[-1]
    if return_snapshots:
        return final, step_snapshots
    return final
