"""Latent DDNM v3 -- instrumented, with the data-consistency form made explicit.

v1 (functions/latent_ddnm.py) hard-codes ONE data-consistency form:

    z0_hat = E( D(z0) - lambda * A^+(A D(z0) - y) )                      [dc_mode="encode"]

That form is the interesting one, because when the DDNM fix A^+(A x0 - y) goes to
zero the step degenerates to a bare latent round-trip  z0_hat = E(D(z0)).  Measured
on this SPNN-512 / SD1.5-VAE pair (exp/v3/diag_mechanism.json):

    ||E(D(z)) - z|| / ||z||   per cycle:   SPNN 0.00054      VAE 0.1293
    image PSNR after 20 latent cycles  :   SPNN 31.23 dB     VAE 11.23 dB

So an invertible codec makes that degenerate step a no-op and a lossy one does not.
Whether a given restoration task *exposes* that gap depends on how much of the image
the DDNM fix re-pins from the measurement each step: whatever A pins is reset every
step and cannot drift, and only the NULL space accumulates codec error.  Hence the
per-step diagnostics below -- they report the fix size, the bare cycle error, and how
much of each lands in the null space.

Also provided: `dc_mode="residual"`, z0 + (E(x0_hat) - E(x0)), which cancels the
codec's cycle error by construction (it is a difference of two encodes).  It costs a
second encode per step and it makes the two codecs behave identically -- it is here
as the control that shows the gap is really about cycle consistency, not as the
recommended setting.

Nothing here clamps the decoded image: both decoders overshoot [-1,1] (SPNN to
+2.26, VAE to +1.35) and clamping inside the loop is a non-invertible operation that
would destroy the fixed point the whole argument rests on.
"""

import math

import torch
import torch.nn.functional as F

from functions.latent_ddnm import (  # unchanged pieces, imported rather than copied
    sd15_alphas_cumprod,
    make_alpha_bar,
    make_inpaint_mask,
    load_mask_file,
    get_schedule_jump,
)

__all__ = [
    "sd15_alphas_cumprod", "make_alpha_bar", "make_inpaint_mask", "load_mask_file",
    "harmonic_extend", "make_pinv_inpaint", "make_pinv_sr", "make_pinv_colorization",
    "make_timesteps",
    "mean_upsample", "color2gray", "lambda_schedule", "latent_ddnm_sample_v3",
]


# --------------------------------------------------------------------------------------
# right inverses.  A^+ only has to satisfy A A^+ A = A; off the range of A it is free.
# --------------------------------------------------------------------------------------

def _jacobi_blur(u):
    """One Jacobi sweep of the 5-point Laplacian (Neumann at the image border)."""
    k = torch.tensor([[0., 1., 0.], [1., 0., 1.], [0., 1., 0.]], device=u.device,
                     dtype=u.dtype) / 4.0
    k = k.view(1, 1, 3, 3).expand(u.shape[1], 1, 3, 3)
    return F.conv2d(F.pad(u, (1, 1, 1, 1), mode="replicate"), k, groups=u.shape[1])


def harmonic_extend(r, m, iters=48, min_size=4):
    """Extend r off the support m with minimum Dirichlet energy (multigrid-lite).

    Returns u with u == r where m == 1 and u harmonic where m == 0, so
    `mask * u == mask * r` exactly and A^+ stays an exact right inverse.
    """
    vs, ws = [r * m], [m.expand_as(r) if m.shape[1] == 1 else m]
    while min(vs[-1].shape[-2:]) > min_size:
        vs.append(F.avg_pool2d(vs[-1], 2))
        ws.append(F.avg_pool2d(ws[-1], 2))
    u = vs[-1] / ws[-1].clamp_min(1e-8)
    for lvl in range(len(vs) - 1, -1, -1):
        w = ws[lvl]
        v = vs[lvl] / w.clamp_min(1e-8)
        if u.shape[-2:] != v.shape[-2:]:
            u = F.interpolate(u, size=v.shape[-2:], mode="bilinear", align_corners=False)
        c = w.clamp(0.0, 1.0)
        u = c * v + (1 - c) * u
        for _ in range(iters):
            u = c * v + (1 - c) * _jacobi_blur(u)
    return u


def make_pinv_inpaint(mask, mode="hardcut", iters=48):
    """A = mask * u.  `hardcut` is Moore-Penrose; `harmonic` extends the residual
    smoothly into the hole (still exact on the mask, so A A^+ A = A holds)."""
    if mode == "hardcut":
        return lambda r: mask * r
    if mode == "harmonic":
        return lambda r: harmonic_extend(mask * r, mask, iters=iters)
    raise ValueError(f"unknown inpaint pinv mode: {mode!r}")


def mean_upsample(x, s):
    """Replicate each low-res pixel into an s x s block (DDNM's A^+ for avg-pool)."""
    n, c, h, w = x.shape
    out = torch.zeros(n, c, h, s, w, s, device=x.device, dtype=x.dtype) + x.view(n, c, h, 1, w, 1)
    return out.view(n, c, s * h, s * w)


def make_pinv_sr(A, scale, mode="block"):
    """A = average-pool by `scale`.

    `block`  : MeanUpsample -- exact, but piecewise-constant on s x s blocks.
    `smooth` : bicubic upsample corrected to be exact,
               A^+ r = U(r) + MeanUpsample(r - A(U(r))),  so A A^+ = I still holds.
    """
    if mode == "block":
        return lambda r: mean_upsample(r, scale)
    if mode == "smooth":
        def _pinv(r):
            u = F.interpolate(r, scale_factor=scale, mode="bicubic", align_corners=False)
            return u + mean_upsample(r - A(u), scale)
        return _pinv
    raise ValueError(f"unknown sr pinv mode: {mode!r}")


def color2gray(x):
    """DDNM colorization A: average the channels, kept as 3 equal channels."""
    g = x.mean(dim=1, keepdim=True)
    return g.expand(-1, 3, -1, -1)


def make_pinv_colorization():
    """A^+ for `color2gray`: spread the gray channel back over RGB (A A^+ = I)."""
    return lambda r: r[:, :1].expand(-1, 3, -1, -1).contiguous()


# --------------------------------------------------------------------------------------
# schedules
# --------------------------------------------------------------------------------------

def make_timesteps(T_sampling, num_train_timesteps=1000, kind="uniform"):
    """Descending list of training timesteps to visit.

    'uniform'  t = (T-1)*skip ... 0   -- what v1 does (skip = N // T).
    'quad'     quadratic spacing, which puts more of the SAME number of steps at low
               noise.  That matters here: as a_bar -> 1 the UNet's correction and the
               injected noise both vanish, so the step degenerates to
               z <- E(D(z) - lam*A^+(...)), i.e. a bare latent round-trip.  Reweighting
               the schedule changes WHERE the 100 round trips happen, not how many.
    'cosine'   same idea, gentler.
    """
    import numpy as np
    N = num_train_timesteps
    if kind == "uniform":
        skip = N // T_sampling
        ts = [i * skip for i in range(T_sampling)]
    elif kind == "quad":
        u = np.linspace(0.0, 1.0, T_sampling)
        ts = list(np.unique((u ** 2 * (N - 1)).round().astype(int)))
    elif kind == "cosine":
        u = np.linspace(0.0, 1.0, T_sampling)
        ts = list(np.unique(((1 - np.cos(u * np.pi / 2)) * (N - 1)).round().astype(int)))
    else:
        raise ValueError(f"unknown timestep schedule: {kind!r}")
    return sorted(int(t) for t in ts)[::-1]          # descending


def lambda_schedule(mode, step_count, total_steps, at_next, val=1.0, floor=0.0, rate=5.0):
    """Data-consistency strength lambda_t (same shapes as v1)."""
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
    elif mode == "ramp":            # 0 -> val: release DC early, enforce it late
        curve = progress
        return floor + (val - floor) * curve
    else:                            # linear
        curve = 1.0 - progress
    return floor + (val - floor) * curve


# --------------------------------------------------------------------------------------
# sampler
# --------------------------------------------------------------------------------------

@torch.no_grad()
def latent_ddnm_sample_v3(unet, codec, alpha_bar, context, A, Ap, y,
                          sigma_y=0.0, T_sampling=100, num_train_timesteps=1000,
                          travel_length=1, travel_repeat=1, eta=0.85,
                          lambda_mode="const", lambda_val=1.0, lambda_floor=0.0,
                          lambda_rate=5.0, damping_floor=1.0,
                          dc_mode="encode", ddim_form="ddnm", anneal_kappa=1.0,
                          guidance_scale=1.0, context_uncond=None,
                          latent_shape=(1, 4, 64, 64), device="cuda",
                          generator=None, record_hook=None,
                          timesteps=None,
                          diag=False, diag_null_mask=None,
                          return_diag=False):
    """Latent DDNM. One decode + one encode per step (dc_mode='encode').

    dc_mode:
      'encode'   z0_hat = E(x0 - lam*A^+(A x0 - y))            1 decode, 1 encode.
                 Degenerates to z0_hat = E(D(z0)) as the fix -> 0, so codec cycle
                 consistency is what decides the outcome.  This is the v1 form.
      'residual' z0_hat = z0 + (E(x0_hat) - E(x0))             1 decode, 2 encodes.
                 The codec's cycle error cancels between the two encodes, so both
                 codecs behave alike.  Control arm only.

    ddim_form:
      'ref'      the transition this repo's own pixel-space DDNM uses
                 (DDNM-main/guided_diffusion/diffusion.py): sigma_t = sqrt(1-a_t/a_next),
                 c2 = sqrt(1-a_next-sigma_t^2).  Correct DDIM marginal.  Ignores eta.
      'ddnm'     what v1 ported: BOTH the noise and the direction term are scaled by
                 gamma_t = sqrt(1-a_next^2), giving total variance (1-a_next^2)(1-a_next)
                 instead of (1-a_next).  Kept only to reproduce v1.
      'standard' textbook DDIM, sigma = eta*sqrt((1-a_prev)/(1-a_t))*sqrt(1-a_t/a_prev).
      'anneal'   gamma = (1-a_next^2)^(kappa/2) scaling of the noise+direction terms.
                 kappa=0 is the correct DDIM marginal, kappa=1 is v1, larger kappa drops
                 into the projection/fixed-point regime sooner.  See the branch below.

    diag=True additionally records, per step (costs one extra encode per step, so it
    is off by default and must not be used for timing):
      res_rms   ||A x0 - y||           how far the current estimate violates the data
      fix_rms   ||lam * A^+(A x0 - y)||  the DDNM fix -- the thing that should -> 0
      cyc_rms   ||E(D(z0)) - z0||      the bare latent round-trip error
      dz_rms    ||z0_hat - z0||        the actual latent change the step applies
      null_frac fraction of the fix energy lying in the null space of A
    """
    skip = num_train_timesteps // T_sampling
    n = latent_shape[0]
    z0_preds = []
    zs = [torch.randn(latent_shape, device=device, generator=generator)]

    if timesteps is None:
        # v1 path: uniform grid, optional RePaint time-travel.  Indices are in units
        # of `skip` and get multiplied back up inside the loop.
        times = get_schedule_jump(T_sampling, travel_length, travel_repeat)
        time_pairs = list(zip(times[:-1], times[1:]))
        scale_t = skip
    else:
        # explicit descending training timesteps; no time travel.
        if travel_length != 1 or travel_repeat != 1:
            raise ValueError("time travel (travel_length/travel_repeat) is only supported on "
                             "the uniform schedule; pass timesteps=None to use it")
        ts = list(timesteps) + [-1]
        time_pairs = list(zip(ts[:-1], ts[1:]))
        scale_t = 1
    total_steps = sum(1 for a, b in time_pairs
                      if (b * scale_t if b >= 0 else -1) < a * scale_t)
    step_count = 0
    do_cfg = (guidance_scale != 1.0) and (context_uncond is not None)
    log = {k: [] for k in ("t", "res_rms", "fix_rms", "cyc_rms", "dz_rms", "null_frac", "lam")}

    def _rms(v):
        return v.pow(2).mean().sqrt().item()

    for i, j in time_pairs:
        i, j = i * scale_t, j * scale_t
        if j < 0:
            j = -1

        if j < i:
            step_count += 1
            t = torch.full((n,), i, device=device, dtype=torch.long)
            at = alpha_bar[i + 1].view(1, 1, 1, 1)
            at_next = alpha_bar[j + 1].view(1, 1, 1, 1)
            zt = zs[-1].to(device)

            if do_cfg:
                z_in, t_in = torch.cat([zt, zt], 0), torch.cat([t, t], 0)
                c_in = torch.cat([context_uncond, context], 0)
                et_u, et_c = unet(z_in, t_in, encoder_hidden_states=c_in).sample.chunk(2)
                et = et_u + guidance_scale * (et_c - et_u)
            else:
                et = unet(zt, t, encoder_hidden_states=context).sample

            z0_t = (zt - et * (1 - at).sqrt()) / at.sqrt()

            lambda_t = lambda_schedule(lambda_mode, step_count, total_steps, at_next,
                                       lambda_val, lambda_floor, lambda_rate)

            # ---- data consistency: the only place pixels appear ----
            x0_t = codec.decode(z0_t)                      # 1 decode
            residual = A(x0_t) - y
            fix = lambda_t * Ap(residual)
            x0_t_hat = x0_t - fix

            # E(x0_t) is needed by the 'residual' mode and by the diagnostics; compute it
            # at most once so neither one adds a round trip the other already paid for.
            z_cyc = codec.encode(x0_t) if (diag or dc_mode == "residual") else None

            if dc_mode == "encode":
                z0_t_hat = codec.encode(x0_t_hat)          # 1 encode
            elif dc_mode == "residual":
                z0_t_hat = z0_t + (codec.encode(x0_t_hat) - z_cyc)
            else:
                raise ValueError(f"unknown dc_mode: {dc_mode!r}")

            if diag:
                log["t"].append(i)
                log["lam"].append(float(lambda_t))
                log["res_rms"].append(_rms(residual))
                log["fix_rms"].append(_rms(fix))
                log["cyc_rms"].append(_rms(z_cyc - z0_t))
                log["dz_rms"].append(_rms(z0_t_hat - z0_t))
                if diag_null_mask is not None:
                    log["null_frac"].append(_rms(fix * diag_null_mask) / max(_rms(fix), 1e-12))

            # ---- transition ----
            noise = torch.randn(latent_shape, device=device, generator=generator)
            if ddim_form == "ref":
                # Exactly DDNM-main/guided_diffusion/diffusion.py:419,460,463 -- the
                # pixel-space DDNM this repo already validated.  sigma_t^2 + c2^2 =
                # 1 - a_next, i.e. the correct DDIM marginal (v1's port did not).
                sigma_t = (1 - at / at_next).clamp_min(0).sqrt()
                c2 = (1 - at_next - sigma_t ** 2).clamp_min(0).sqrt()
                damp = 1.0 - at_next.item() * (1.0 - damping_floor)
                zt_next = at_next.sqrt() * z0_t_hat + damp * c2 * et + sigma_t * noise
            elif ddim_form == "standard":
                var = ((1 - at_next) / (1 - at)).clamp_min(0) * (1 - at / at_next).clamp_min(0)
                sigma = eta * var.sqrt()
                dir_coef = (1 - at_next - sigma ** 2).clamp_min(0).sqrt()
                zt_next = at_next.sqrt() * z0_t_hat + dir_coef * et + sigma * noise
            elif ddim_form == "anneal":
                # One-parameter family interpolating "diffusion step" -> "projection step".
                #
                #   z_next = sqrt(a_next) z0_hat + gamma * sqrt(1-a_next) * (eta*n + sqrt(1-eta^2)*et)
                #   gamma  = (1 - a_next^2) ** (kappa/2)
                #
                # kappa=0 -> gamma=1 -> total variance (1-a_next): the CORRECT DDIM marginal.
                # kappa=1 -> exactly what v1 ported.
                # As a_bar -> 1, gamma -> 0 like (1-a_next)^(kappa/2), so the step collapses to
                #       z <- z0_hat = E(D(z) - lam*A^+(A D(z) - y)),
                # an alternating projection between the measurement set and the codec's range.
                # That iteration converges only if E(D(.)) has a fixed point -- which is exactly
                # what separates an invertible codec from a lossy one.  kappa therefore controls
                # how much of the sampler is fixed-point iteration, and the codec gap tracks it.
                gamma_t = (1 - at_next ** 2).clamp_min(0) ** (anneal_kappa / 2.0)
                c1 = (1 - at_next).sqrt() * eta
                c2 = (1 - at_next).sqrt() * ((1 - eta ** 2) ** 0.5)
                damp = 1.0 - at_next.item() * (1.0 - damping_floor)
                zt_next = at_next.sqrt() * z0_t_hat + gamma_t * (c1 * noise + damp * c2 * et)
            else:  # 'ddnm' -- official DDNM transition, as ported in v1
                sigma_t = (1 - at_next ** 2).sqrt()
                if sigma_t >= at_next * sigma_y:
                    gamma_t = (sigma_t ** 2 - (at_next * sigma_y) ** 2).sqrt()
                else:
                    gamma_t = torch.zeros_like(sigma_t)
                c1 = (1 - at_next).sqrt() * eta
                c2 = (1 - at_next).sqrt() * ((1 - eta ** 2) ** 0.5)
                damp = 1.0 - at_next.item() * (1.0 - damping_floor)
                zt_next = at_next.sqrt() * z0_t_hat + gamma_t * (c1 * noise + damp * c2 * et)

            z0_preds.append(z0_t)
            zs.append(zt_next)
            if record_hook is not None:
                record_hook(i, x0_t, x0_t_hat)
        else:  # RePaint time-travel back
            at_next = alpha_bar[j + 1].view(1, 1, 1, 1)
            z0_t = z0_preds[-1]
            noise = torch.randn(latent_shape, device=device, generator=generator)
            zs.append(at_next.sqrt() * z0_t + noise * (1 - at_next).sqrt())

    x_final = codec.decode(zs[-1])
    if return_diag:
        return x_final, log
    return x_final
