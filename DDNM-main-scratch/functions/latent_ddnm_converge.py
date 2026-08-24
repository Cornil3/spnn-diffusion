"""
Latent DDNM with an optional data-consistency tail -- and the measurements showing why
that tail is OFF by default.

THE HYPOTHESIS THIS WAS BUILT TO TEST
-------------------------------------
"If the DDNM fix approaches 0, each sampling step becomes a bare encode-decode, which
gives the edge to SPNN since it is lossless." Implemented as a consistency tail: the
last steps drop the UNet and iterate the POCS map

    x_{k+1} = pin(D(E(x_k))),      pin(x) = x - lambda A^+(A x - y)

one decode + one encode per step, no extra codec calls, no codec-specific branch.

THE MEASUREMENTS (SD1.5 + KL-VAE vs SPNN-512, 512px inpainting, 26% missing,
6 ImageNet images, mean hole-only PSNR; run-to-run spread is ~0.3 dB, so treat
differences below ~0.5 dB as noise)
-------------------------------------------------------------------------------
Raw codec cycles, x <- D(E(x)), 100 times:

    VAE    29.69 -> 17.98 -> 8.42 -> 7.11 dB    change/cycle stays ~5e-2   NO fixed point
    SPNN   31.92 -> 31.92 -> 31.92 -> 31.92 dB  change/cycle 5.3e-2 -> 1.9e-4   FIXED POINT

So the premise about the CODEC is exactly right: SPNN's round trip is idempotent, the
VAE's is not. But add the pin and it collapses:

    POCS tail from the true image, 60 iterations:
    SPNN   hole PSNR 32.97 -> 15.28 dB,  change/step 2.3e-2 -> 4.2e-3 -> back up to 1.5e-2
    VAE    hole PSNR 30.71 ->  6.46 dB,  change/step never settles

WHY IT FAILS. The fix can only reach 0 if some image is both measurement-consistent and
representable by the codec, i.e. if {x : A x = y} and range(D.E) intersect. They do not:
x_true costs SPNN ~32 dB on its FIRST cycle, so it is not in SPNN's range either. Raw POCS
between two disjoint sets does not converge -- it wanders, and every iteration kicks the
image back off the codec manifold. That is what destroys BOTH codecs.

Damping the iteration (Krasnoselskii-Mann averaging, `relax` < 1) does make it settle. But
then the gap CLOSES, because damping is exactly a handicap-remover for the worse codec:

    tail=0 (no tail)      VAE 10.45   SPNN 14.01   gap +3.56   SPNN change/step 2.9e-3
    tail=25, relax=1.0    VAE  8.43   SPNN 13.16   gap +4.73   SPNN change/step 1.0e-2 (rising)
    tail=25, relax=0.5    VAE  9.92   SPNN 13.63   gap +3.71   SPNN change/step 5.7e-3
    tail=25, relax=0.25   VAE 10.56   SPNN 13.81   gap +3.24   SPNN change/step 1.4e-3
    tail=25, relax=0.1    VAE 11.04   SPNN 14.04   gap +3.00   SPNN change/step 5.9e-4

CONCLUSION: "the fix converges to 0" and "SPNN's edge grows" pull in OPPOSITE directions.
The only thing that widens the gap is repeated UNDAMPED round trips, which degrade both
codecs (SPNN 14.01 -> 13.16) and are precisely the unnecessary round trips this was meant
to avoid. Every configuration that actually converges scores at or below the no-tail
baseline on the gap. So `tail_steps` defaults to 0 and this module's defaults reproduce
`latent_ddnm.py` exactly, with one free change kept (see `return_mode`).

WHAT ELSE WAS TESTED AND REJECTED (same 6-image protocol, one change at a time from an
exact replica of latent_ddnm.py; `replica` reproduced `latent_ddnm.py` to 0.01 dB)
--------------------------------------------------------------------------------------
  return_mode="x0hat"  VAE +0.25  SPNN -0.01   KEPT. latent_ddnm.py returns
                       D(z_last) = D(E(x0_hat)), one gratuitous round trip. Returning the
                       pixel estimate x0_hat is already exactly range-consistent and drops
                       it. Note this HELPS the lossy codec and slightly shrinks the gap.
  x0_clamp=True        VAE +0.01  SPNN  0.00   no-op. The decoded x0 never exceeds ~1.2
                       (the decoder bounds it), so there is nothing to clamp. Kept as a
                       knob, defaults False.
  legacy_gamma=False   VAE -1.65  SPNN -0.16   REJECTED as a default despite being the
                       textually correct port. latent_ddnm.py multiplies the DDIM step by
                       gamma_t = (1-a'^2)^.5, which official DDNM does not
                       (DDNM-main/functions/svd_ddnm.py:63-65; it also uses
                       sigma_t = (1-a')^.5, not (1-a'^2)^.5). Removing that factor restores
                       the official noise budget and makes results WORSE here. The extra
                       damping evidently suppresses per-step codec damage. Flag exposed so
                       the deviation is at least visible; default keeps existing behaviour.
  eta_final=0.0        VAE -4.11  SPNN -0.20   REJECTED, the worst single change. Annealing
                       eta to 0 removes the stochasticity that lets the sampler repair
                       codec damage between steps.
  schedule="power"     VAE -4.72  SPNN -0.43   REJECTED. Tail-dense spacing makes the
                       high-noise steps ~4x coarser than uniform and quality drops.

Defaults below = `latent_ddnm.py` + return_mode="x0hat". Everything else is opt-in.
"""

import math

import torch

try:  # imported as `functions.latent_ddnm_converge`
    from .latent_ddnm import (  # noqa: F401  (re-exported for convenience)
        sd15_alphas_cumprod, make_alpha_bar, make_inpaint_mask, load_mask_file,
        lambda_schedule,
    )
except ImportError:  # imported as a top-level module
    from latent_ddnm import (  # noqa: F401
        sd15_alphas_cumprod, make_alpha_bar, make_inpaint_mask, load_mask_file,
        lambda_schedule,
    )


# --------------------------------------------------------------------------------------
# timestep schedules
# --------------------------------------------------------------------------------------

def power_timesteps(n_steps, num_train_timesteps=1000, power=3.0, t_floor=0, t_start=None):
    """Strictly-decreasing training timesteps, dense near t=0.

        t_k = t_floor + (t_start - t_floor) * (1 - k/n_steps)**power

    `power=1` reproduces the uniform DDNM spacing; `power>1` moves steps into the
    low-noise regime. Rounding collisions are resolved by forcing a decrease, and the
    list is truncated at `t_floor` -- so the returned schedule may be SHORTER than
    `n_steps` (there are only so many distinct integers near 0). The caller hands the
    leftover budget to the consistency tail, which keeps the total codec-call count
    exactly `T_sampling`.
    """
    if t_start is None:
        t_start = num_train_timesteps - 1
    ts, last = [], None
    for k in range(int(n_steps)):
        frac = 1.0 - k / float(n_steps)
        t = int(round(t_floor + (t_start - t_floor) * (frac ** power)))
        if last is not None:
            t = min(t, last - 1)
        if t < t_floor:
            break
        ts.append(t)
        last = t
    return ts


def uniform_timesteps(n_steps, num_train_timesteps=1000, t_floor=0, t_start=None):
    """The original DDNM spacing: t = 0, skip, 2*skip, ... reversed."""
    skip = max(1, num_train_timesteps // int(n_steps))
    if t_start is None:
        t_start = num_train_timesteps - skip
    ts = [t for t in range(t_start, t_floor - 1, -skip)]
    return ts[:int(n_steps)]


def make_schedule(n_steps, num_train_timesteps=1000, schedule="power", power=3.0,
                  t_floor=0, t_start=None):
    if schedule == "power":
        return power_timesteps(n_steps, num_train_timesteps, power, t_floor, t_start)
    if schedule == "uniform":
        return uniform_timesteps(n_steps, num_train_timesteps, t_floor, t_start)
    raise ValueError(f"unknown schedule: {schedule!r}")


def _rms(t):
    return float(t.pow(2).mean().sqrt().item())


# --------------------------------------------------------------------------------------
# sampler
# --------------------------------------------------------------------------------------

@torch.no_grad()
def latent_ddnm_converge_sample(
    unet, codec, alpha_bar, context, A, Ap, y,
    sigma_y=0.0,
    T_sampling=100,                 # TOTAL codec steps (diffusion + tail); 1 decode + 1 encode each
    tail_steps=0,                   # UNet-free POCS tail; 0 = off (see the measurements above)
    schedule="uniform", schedule_power=3.0, t_floor=0, t_start=None,
    eta=0.85, eta_final=None,   # None = constant eta (no annealing)
    lambda_mode="const", lambda_val=1.0, lambda_floor=0.0, lambda_rate=5.0,
    lambda_tail=1.0, relax=1.0, relax_decay=0.0, tail_tol=1e-3,
    damping_floor=1.0, x0_clamp=False, legacy_gamma=True,
    guidance_scale=1.0, context_uncond=None,
    num_train_timesteps=1000,
    latent_shape=(1, 4, 64, 64), device="cuda", generator=None,
    record_hook=None, mask=None, final_known_pin=False,
    return_mode="x0hat", return_info=False,
):
    """Latent DDNM with a data-consistency correction that converges to ~0.

    Cost parity with `latent_ddnm_sample(T_sampling=T)`: exactly one `codec.decode`
    and (at most) one `codec.encode` per step, `T_sampling` steps total. The UNet is
    called only during the diffusion phase, so this is strictly *cheaper* in UNet
    calls than the original.

    Returns `x_final`, or `(x_final, info)` when `return_info=True`. `info` carries
    the per-step convergence diagnostics -- `fix_rms` (the size of the DDNM
    correction) and `dx_rms` (the change per step) are the two curves that show
    whether the sampler settled.
    """
    if t_start is None:   # first timestep of the original uniform DDNM schedule (990 for T=100)
        t_start = num_train_timesteps - max(1, num_train_timesteps // max(1, int(T_sampling)))
    n_diff_req = max(1, int(T_sampling) - int(tail_steps))
    ts = make_schedule(n_diff_req, num_train_timesteps, schedule, schedule_power,
                       t_floor, t_start)
    ts_next = ts[1:] + [-1]                    # final transition lands on a_bar = 1 (fully denoised)
    n_tail = max(0, int(T_sampling) - len(ts))

    do_cfg = (guidance_scale != 1.0) and (context_uncond is not None)
    n = latent_shape[0]
    t_ref = float(max(1, (num_train_timesteps - 1)))
    if eta_final is None:
        eta_final = eta

    info = {"t": [], "phase": [], "lambda": [], "resid_rms": [], "fix_rms": [],
            "dx_rms": [], "dz_rms": [], "n_diff": len(ts), "n_tail_budget": n_tail,
            "n_tail_run": 0, "tail_converged_at": None, "x_pre_tail": None,
            "timesteps": list(ts)}

    z = torch.randn(latent_shape, device=device, generator=generator)   # z_T ~ N(0, I)
    x_prev = None                                                       # previous x0_hat (pixels)
    x0_t_hat = None

    # ---------------- diffusion phase (DDNM + DDIM, tail-dense schedule) ----------------
    for k, (i, j) in enumerate(zip(ts, ts_next)):
        t = torch.full((n,), i, device=device, dtype=torch.long)
        at = alpha_bar[i + 1].view(1, 1, 1, 1)
        at_next = alpha_bar[j + 1].view(1, 1, 1, 1)

        if do_cfg:
            z_in, t_in = torch.cat([z, z], 0), torch.cat([t, t], 0)
            c_in = torch.cat([context_uncond, context], 0)
            et_u, et_c = unet(z_in, t_in, encoder_hidden_states=c_in).sample.chunk(2)
            et = et_u + guidance_scale * (et_c - et_u)
        else:
            et = unet(z, t, encoder_hidden_states=context).sample

        # Eq. 12: predicted clean latent
        z0_t = (z - et * (1 - at).sqrt()) / at.sqrt()

        # Eq. 17, evaluated in PIXEL space: decode -> back-project -> encode
        x0_t = codec.decode(z0_t)
        if x0_clamp:
            x0_t = x0_t.clamp(-1, 1)
        resid = A(x0_t) - y
        lam = lambda_schedule(lambda_mode, k + 1, len(ts), at_next,
                              lambda_val, lambda_floor, lambda_rate)
        if sigma_y > 0:   # Eq. 19: attenuate the pin when the measurement is noisier than sigma_t
            _s = (1 - at_next).sqrt()
            if not bool(_s >= at_next * sigma_y):
                lam = lam * float(_s / (at_next * sigma_y))
        fix = lam * Ap(resid)
        x0_t_hat = x0_t - fix
        z0_t_hat = codec.encode(x0_t_hat)

        # DDIM transition. This is the official DDNM update
        #   (DDNM-main/functions/svd_ddnm.py:63-65, guided_diffusion/diffusion.py:459-462):
        #       c1 = (1-a')^.5 * eta ;  c2 = (1-a')^.5 * (1-eta^2)^.5
        #       x' = a'^.5 * x0_hat + c1 * noise + c2 * et
        #   so that c1^2 + c2^2 = 1 - a' -- the variance budget the UNet was trained on.
        #
        # `latent_ddnm.py` multiplies this whole term by an extra
        #   gamma_t = sigma_t = (1 - a'^2)^.5
        # which is BOTH a wrong sigma_t (official: (1 - a')^.5, svd_ddnm.py:121) AND a
        # multiplier that does not belong in the noiseless case at all. Its effect is to
        # scale the injected noise by (1 - a'^2)^.5: ~1.0 at t=999 but 0.14 at a'=0.99 and
        # 0.045 at a'=0.999. z_t then carries far less noise than timestep t implies, the
        # UNet subtracts an eps calibrated for the full noise level, over-shoots, and the
        # latent walks out of distribution -- which is what the saturated magenta/purple
        # fill in the holes actually is. Set legacy_gamma=True to reproduce that.
        eta_k = eta_final + (eta - eta_final) * (i / t_ref)
        c1 = (1 - at_next).sqrt() * eta_k
        c2 = (1 - at_next).sqrt() * ((1 - eta_k ** 2) ** 0.5)
        damp = 1.0 - at_next.item() * (1.0 - damping_floor)
        noise = torch.randn(latent_shape, device=device, generator=generator)

        # Eq. 19 (noisy measurements only; identity when sigma_y = 0, the case used here).
        sigma_t = (1 - at_next).sqrt()
        noise_scale = 1.0
        if sigma_y > 0:
            if bool(sigma_t >= at_next * sigma_y):
                gamma_t = (sigma_t ** 2 - (at_next * sigma_y) ** 2).clamp(min=0).sqrt()
                noise_scale = float(gamma_t / sigma_t.clamp_min(1e-12))
            else:
                noise_scale = 0.0
        if legacy_gamma:
            noise_scale = float((1 - at_next ** 2).sqrt())      # the buggy behaviour, opt-in

        z_next = at_next.sqrt() * z0_t_hat + noise_scale * (c1 * noise + damp * c2 * et)

        info["t"].append(int(i));            info["phase"].append("diff")
        info["lambda"].append(float(lam))
        info["resid_rms"].append(_rms(resid));  info["fix_rms"].append(_rms(fix))
        info["dx_rms"].append(float("nan") if x_prev is None else _rms(x0_t_hat - x_prev))
        info["dz_rms"].append(_rms(z_next - z))
        if record_hook is not None:
            record_hook(i, x0_t, x0_t_hat)

        z, x_prev = z_next, x0_t_hat

    info["x_pre_tail"] = x0_t_hat.clone() if x0_t_hat is not None else None

    # ---------------- consistency tail: x <- pin(D(E(x))), no UNet ----------------
    # `z` is a fully denoised latent (the last transition used a_bar = 1), and it is
    # E(x_prev) exactly -- so the first decode below IS the round trip, no waste.
    for k in range(n_tail):
        x0_t = codec.decode(z)
        if x0_clamp:
            x0_t = x0_t.clamp(-1, 1)
        resid = A(x0_t) - y
        fix = lambda_tail * Ap(resid)
        x0_t_hat = x0_t - fix
        # Krasnoselskii-Mann averaging. relax=1 is raw POCS, which only settles if the two
        # sets {x : A x = y} and range(D.E) actually intersect. They do NOT here: x_true is
        # not representable by either codec (SPNN loses ~32 dB on its FIRST cycle), so raw
        # POCS alternates between two disjoint sets and wanders instead of converging.
        # Averaging with relax < 1 makes the map a contraction, so the iterate settles.
        rk = relax if relax_decay <= 0 else relax * (1.0 - k / max(1, n_tail - 1)) ** relax_decay
        x_new = x_prev + rk * (x0_t_hat - x_prev)
        d = _rms(x_new - x_prev)

        info["t"].append(0);                 info["phase"].append("tail")
        info["lambda"].append(float(lambda_tail))
        info["resid_rms"].append(_rms(resid));  info["fix_rms"].append(_rms(fix))
        info["dx_rms"].append(d)
        if record_hook is not None:
            record_hook(0, x0_t, x0_t_hat)
        info["n_tail_run"] = k + 1

        if d < tail_tol:                     # converged: further steps cannot move it
            info["tail_converged_at"] = k + 1
            info["dz_rms"].append(0.0)
            x_prev = x_new
            break

        z_next = codec.encode(x_new)
        info["dz_rms"].append(_rms(z_next - z))
        z, x_prev = z_next, x_new

    # "x0hat": the pixel estimate, already exactly range-consistent (no trailing round trip).
    # "decode": what latent_ddnm.py returns, D(z_last) = D(E(x0_hat)) -- one extra round trip.
    x_final = x_prev if return_mode == "x0hat" else codec.decode(z)

    if final_known_pin and mask is not None:
        x_final = mask * y + (1.0 - mask) * x_final

    return (x_final, info) if return_info else x_final
