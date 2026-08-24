"""
Latent-space DDNM (Denoising Diffusion Null-space Model).

The latent generalization of the official pixel-space DDNM update. Pixel DDNM,
per reverse step:

    x0      = (x_t - sqrt(1-a_t) * eps) / sqrt(a_t)        # predicted clean image
    x0_hat  = x0 - A^+(A x0 - y)                           # pin range space to y
    x_{t-1} = sqrt(a_{t-1}) x0_hat + (DDIM noise terms)

Here the diffusion runs in a latent space z (SD1.5's) while the degradation A and
measurement y live in pixel space, so we bridge them with a codec (E=encode,
D=decode):

    z0      = (z_t - sqrt(1-a_t) * eps) / sqrt(a_t)        # predicted clean latent
    x0      = D(z0)                                        # decode to pixels
    x0_hat  = x0 - A^+(A x0 - y)                           # SAME null-space step, in pixels
    z0_hat  = E(x0_hat)                                    # re-encode
    z_{t-1} = sqrt(a_{t-1}) z0_hat + (DDIM noise terms)

The per-step decode -> pin -> encode round-trip is exact data consistency in
pixel space but leans entirely on E(D(.)) ~= identity. A lossy codec (KL-VAE)
corrupts z0_hat every step and the error compounds; a near-invertible codec
(SPNN) does not.
"""

import numpy as np
import torch


def sd15_alphas_cumprod(num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012,
                        device="cpu"):
    """SD1.x/2.x 'scaled_linear' schedule -> alphas_cumprod [num_train_timesteps].

        betas = linspace(sqrt(b0), sqrt(bT), N) ** 2
        alpha_bar_k = prod_{s<=k} (1 - beta_s)

    Identical to diffusers DDPMScheduler(beta_schedule='scaled_linear', ...), so the
    SD1.5 UNet's epsilon predictions stay calibrated -- computed directly so we never
    depend on the scheduler config file (which is not always in the local HF cache).
    """
    betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5,
                           num_train_timesteps, dtype=torch.float64) ** 2
    alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
    return alphas_cumprod.to(device=device, dtype=torch.float32)


def make_alpha_bar(alphas_cumprod):
    """Prepend 1.0 so that index (t+1) maps to a_bar(t).

    Mirrors the official DDNM `compute_alpha`: timestep t in [0, N-1] reads
    alpha_bar[t+1]; the sentinel t = -1 (final, fully denoised) reads index 0 = 1.
    """
    one = torch.ones(1, dtype=alphas_cumprod.dtype, device=alphas_cumprod.device)
    return torch.cat([one, alphas_cumprod], dim=0)


def make_inpaint_mask(image_size, kind="box", box_frac=0.5, device="cpu", generator=None):
    """Binary mask [1, 1, H, W]; 1 = known/observed pixel, 0 = hole to inpaint."""
    H = W = int(image_size)
    mask = torch.ones(1, 1, H, W, device=device)
    if kind == "box":
        s = int(round(box_frac * image_size))
        top, left = (H - s) // 2, (W - s) // 2
        mask[..., top:top + s, left:left + s] = 0.0
    elif kind == "half":
        mask[..., :, W // 2:] = 0.0
    elif kind == "random":
        keep = (torch.rand(1, 1, H, W, device=device, generator=generator) > box_frac).float()
        mask = keep
    else:
        raise ValueError(f"unknown mask kind: {kind!r}")
    return mask


def load_mask_file(path, image_size, device="cpu"):
    """Load a mask from .npy or an image file. 1 = known, 0 = hole."""
    if path.endswith(".npy"):
        m = torch.from_numpy(np.load(path)).float()
    else:
        from PIL import Image
        im = Image.open(path).convert("L")
        m = (torch.from_numpy(np.array(im)).float() / 255.0 > 0.5).float()
    m = m.view(1, 1, *m.shape[-2:])
    if m.shape[-1] != image_size or m.shape[-2] != image_size:
        m = torch.nn.functional.interpolate(m, size=(image_size, image_size), mode="nearest")
    return m.to(device)


def lambda_schedule(mode, step_count, total_steps, at_next, val=1.0, floor=0.0, rate=5.0):
    """Data-consistency strength lambda_t over sampling (matches latent_diffusion.py).

    const  : val (flat).
    exp    : decay val -> floor, exp tempo `rate` (>0).
    linear : decay val -> floor, linearly.
    noise  : floor + (val-floor)*(1 - a_next)  (strong early).
    """
    import math
    if mode == "const":
        return val
    if mode == "noise":
        return floor + (val - floor) * float(1.0 - at_next.item())
    progress = min(1.0, (step_count - 1) / max(1, total_steps - 1))
    if mode == "exp":
        if rate > 0.0:
            e1 = math.exp(-rate)
            curve = (math.exp(-rate * progress) - e1) / (1.0 - e1)
        else:
            curve = 1.0 - progress
    else:  # linear
        curve = 1.0 - progress
    return floor + (val - floor) * curve


@torch.no_grad()
def latent_ddnm_sample(unet, codec, alpha_bar, context, A, Ap, y,
                       sigma_y=0.0, T_sampling=100, num_train_timesteps=1000,
                       travel_length=1, travel_repeat=1, eta=0.85,
                       lambda_mode="const", lambda_val=1.0, lambda_floor=0.0,
                       lambda_rate=5.0, damping_floor=1.0,
                       guidance_scale=1.0, context_uncond=None,
                       latent_shape=(1, 4, 64, 64), device="cuda",
                       generator=None, record_hook=None,
                       mask=None, final_known_pin=True):
    """Latent-space port of Diffusion.simplified_ddnm_plus (guided_diffusion/diffusion.py).

    The reverse loop is the pixel-space DDNM+ loop, unchanged except for FOUR things:
      (1) the diffused variable is the latent z, not the pixel image x;
      (2) model(xt, t)                  ->  unet(zt, t, context)   (+ optional CFG,
          since SD1.5's UNet is conditional; pass the empty-prompt embedding);
      (3) the back-projection (bp) line is wrapped by the codec -- the only place
          pixels appear:   x0 = decode(z0);  x0_hat = x0 - lambda*Ap(A x0 - y);
                           z0_hat = encode(x0_hat);
      (4) the returned sample is decode(z).
    Everything else -- lambda_t/gamma_t (Eq. 19), the DDIM step, the RePaint
    time-travel schedule -- is identical to the original. `alpha_bar` is
    alphas_cumprod with a leading 1.0, so alpha_bar[t+1] == compute_alpha(betas, t).

    Args mirror the original; `sigma_y` (measurement-noise std) and
    `travel_length/travel_repeat` (time-travel) default to the noise-free, no-travel
    case. `record_hook(i, x0_t, x0_t_hat)` receives the per-step pixel frames.
    """
    skip = num_train_timesteps // T_sampling
    n = latent_shape[0]
    z0_preds = []
    zs = [torch.randn(latent_shape, device=device, generator=generator)]  # z_T ~ N(0, I)

    times = get_schedule_jump(T_sampling, travel_length, travel_repeat)
    time_pairs = list(zip(times[:-1], times[1:]))
    total_steps = sum(1 for a, b in time_pairs
                      if (b * skip if b >= 0 else -1) < a * skip)
    step_count = 0
    do_cfg = (guidance_scale != 1.0) and (context_uncond is not None)

    for i, j in time_pairs:
        i, j = i * skip, j * skip
        if j < 0:
            j = -1

        if j < i:  # normal sampling
            step_count += 1
            t = torch.full((n,), i, device=device, dtype=torch.long)
            at = alpha_bar[i + 1].view(1, 1, 1, 1)
            at_next = alpha_bar[j + 1].view(1, 1, 1, 1)
            sigma_t = (1 - at_next ** 2).sqrt()
            zt = zs[-1].to(device)

            # (2) predict the noise with the latent UNet (replaces model(xt, t))
            if do_cfg:
                z_in, t_in = torch.cat([zt, zt], 0), torch.cat([t, t], 0)
                c_in = torch.cat([context_uncond, context], 0)
                et_u, et_c = unet(z_in, t_in, encoder_hidden_states=c_in).sample.chunk(2)
                et = et_u + guidance_scale * (et_c - et_u)
            else:
                et = unet(zt, t, encoder_hidden_states=context).sample

            # Eq. 12 (predicted clean latent)
            z0_t = (zt - et * (1 - at).sqrt()) / at.sqrt()

            # Eq. 19 residual-noise scale gamma_t (lambda_t comes from the schedule below)
            if sigma_t >= at_next * sigma_y:
                gamma_t = (sigma_t ** 2 - (at_next * sigma_y) ** 2).sqrt()
            else:
                gamma_t = 0.0
            # data-consistency strength lambda_t (const=1.0 default; or decay schedule)
            lambda_t = lambda_schedule(lambda_mode, step_count, total_steps, at_next,
                                       lambda_val, lambda_floor, lambda_rate)

            # (3) Eq. 17 back-projection, evaluated in PIXEL space:  decode -> bp -> encode
            x0_t = codec.decode(z0_t)
            x0_t_hat = x0_t - lambda_t * Ap(A(x0_t) - y)
            z0_t_hat = codec.encode(x0_t_hat)

            # DDIM transition (DDIM, not DDPM, exactly as in the original)
            c1 = (1 - at_next).sqrt() * eta
            c2 = (1 - at_next).sqrt() * ((1 - eta ** 2) ** 0.5)
            damp = 1.0 - at_next.item() * (1.0 - damping_floor)   # damp the et (direction) term
            noise = torch.randn(latent_shape, device=device, generator=generator)
            zt_next = at_next.sqrt() * z0_t_hat + gamma_t * (c1 * noise + damp * c2 * et)

            z0_preds.append(z0_t)
            zs.append(zt_next)
            if record_hook is not None:
                record_hook(i, x0_t, x0_t_hat)
        else:  # time-travel back (RePaint resampling)
            at_next = alpha_bar[j + 1].view(1, 1, 1, 1)
            z0_t = z0_preds[-1]
            noise = torch.randn(latent_shape, device=device, generator=generator)
            zt_next = at_next.sqrt() * z0_t + noise * (1 - at_next).sqrt()
            zs.append(zt_next)

    # (4) decode the final latent back to pixels
    x_final = codec.decode(zs[-1])
    if final_known_pin and mask is not None:
        # final_known_pin: force the unmasked region EXACTLY to the measurement (post-sampling).
        # Sampling is unchanged; only the known region of the output is rewritten.
        # mask: 1=known, 0=hole; y = A(x_orig) = mask*x_orig (hole keeps the sampled result).
        x_final = mask * y + (1.0 - mask) * x_final
    return x_final


# RePaint time-travel schedule (copied verbatim from functions/svd_ddnm.py).
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
