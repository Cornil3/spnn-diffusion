"""
Latent DDNM with hole damping and other cheap stabilisers. A drop-in superset of latent_ddnm.latent_ddnm:
with lam_hole=None and the other extras at their defaults the output is bit-identical to it (noise_rule, clip_x0,
record, stop_frac, sandwich_frac, travel_* behave the same).

Budget: at most ONE decode + ONE encode per step, like plain latent DDNM; blend_frac / sandwich_frac use fewer.

Each reverse step:  z0 -> D(z0) -> paste known pixels -> E(.) -> z0_new.  Let delta = E(paste(D z0)) - z0.
  lam_hole   per latent cell  z0_new = z0 + w * delta,  w = lam_hole + (lam - lam_hole) * (known-pixel fraction of the
             cell's 8x8 patch). lam_hole=0: cells fully over the hole keep the UNet's prediction and are never
             re-encoded, so the encoder's response to the paste seam (and, for a non-idempotent codec, its round-trip
             error) cannot leak into the hole. Measured: fixes the blue fills / drift for the SPNN AND the SD-VAE.
  lam, lam_end   uniform damping of delta (lam_end ramps it linearly over the run)
  blend_frac     first fraction of steps: paste the known region in latent space (no decode, no encode)
  sandwich_frac  after this fraction: no decode/encode (plain latent DDIM); one decode at the end
"""
import torch
import torch.nn.functional as F
import latent_ddnm as ld


def _latent_mask(mask, erode=2):
    """64x64 latent cells lying entirely inside the known region, eroded by `erode` cells."""
    m = (F.avg_pool2d(mask.float(), 8) > 0.999).float()
    if erode:
        m = 1 - F.max_pool2d(1 - m, 2 * erode + 1, stride=1, padding=erode)
    return m


@torch.no_grad()
def latent_ddnm_idem(eps_model, betas, codec, A, Ap, y, latent_shape, T_sampling=100, eta=0.85,
                     travel_length=1, travel_repeat=1, generator=None, record=False, stop_frac=1.0,
                     sandwich_frac=1.0, clip_x0=False, noise_rule="ddnm",
                     lam=1.0, lam_end=None, lam_hole=None, blend_frac=0.0, mask=None):
    device, n = y.device, y.shape[0]
    T = betas.shape[0]
    skip = T // T_sampling
    z = torch.randn((n, *latent_shape), device=device, generator=generator)
    times = ld.get_schedule_jump(T_sampling, travel_length, travel_repeat)
    n_steps = sum(1 for a, b in zip(times[:-1], times[1:]) if b < a)
    paste = lambda x: x - Ap(A(x) - y)
    dec = lambda zz: codec.decode(zz).clamp(-1, 1) if clip_x0 else codec.decode(zz)
    if (blend_frac > 0 or lam_hole is not None) and mask is None:
        raise ValueError("blend_frac / lam_hole need the pixel mask")
    m_lat = _latent_mask(mask) if blend_frac > 0 else None
    z_known = codec.encode(y) if blend_frac > 0 else None
    known_frac = F.avg_pool2d(mask.float(), 8) if lam_hole is not None else None
    frames, k_step, z0_hat, x_hat, d_last = [], 0, None, None, None
    for i, j in zip(times[:-1], times[1:]):
        i, j = i * skip, j * skip
        if j < 0:
            j = -1
        t = torch.full((n,), i, device=device, dtype=torch.long)
        t_next = torch.full((n,), j, device=device, dtype=torch.long)
        at_next = ld.compute_alpha(betas, t_next)
        if j < i:
            at = ld.compute_alpha(betas, t)
            eps = eps_model(z, t).float()
            z0_t = (z - eps * (1 - at).sqrt()) / at.sqrt()
            frac = k_step / n_steps
            k_step += 1
            if frac < blend_frac:                                    # latent-space paste: 0 codec calls
                z0_hat, x_hat = m_lat * z_known + (1 - m_lat) * z0_t, None
            elif k_step > sandwich_frac * n_steps:                   # tail: plain latent DDIM, 0 codec calls
                z0_hat, x_hat = z0_t, None
            else:                                                    # 1 decode + 1 encode
                d_last = dec(z0_t)
                x_hat = paste(d_last)
                e = codec.encode(x_hat)
                lam_k = lam if lam_end is None else lam + (lam_end - lam) * frac
                if lam_hole is None and lam_k == 1.0:
                    z0_hat = e                                       # exact plain-DDNM path
                else:
                    w = lam_k if lam_hole is None else lam_hole + (lam_k - lam_hole) * known_frac
                    z0_hat = z0_t + w * (e - z0_t)
            if noise_rule == "ddim":
                sigma = eta * ((1 - at_next) / (1 - at) * (1 - at / at_next)).clamp_min(0).sqrt()
                c1, c2 = sigma, (1 - at_next - sigma ** 2).clamp_min(0).sqrt()
            else:
                c1 = (1 - at_next).sqrt() * eta
                c2 = (1 - at_next).sqrt() * ((1 - eta ** 2) ** 0.5)
            z = at_next.sqrt() * z0_hat + c1 * torch.randn(z.shape, device=device, generator=generator) + c2 * eps
            if record and x_hat is not None:
                frames.append((i, x_hat.clamp(-1, 1).cpu()))
            if k_step >= stop_frac * n_steps:
                break
        else:                                                         # time travel
            z = at_next.sqrt() * z0_hat + torch.randn(z.shape, device=device, generator=generator) * (1 - at_next).sqrt()
    if x_hat is None:                                                 # ended in a codec-free phase: the one final decode
        d_last = dec(z0_hat)
        x_hat = paste(d_last)
    resid = ((A(d_last) - y) ** 2).flatten(1).sum(1) / A(torch.ones_like(d_last)).flatten(1).sum(1)
    return {"x": x_hat.clamp(-1, 1).cpu(), "z": z0_hat.cpu(), "frames": frames, "resid": resid.sqrt().cpu()}
