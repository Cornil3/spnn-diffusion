"""
Latent DDNM: DDNM (Wang et al., "Zero-Shot Image Restoration Using Denoising Diffusion Null-Space Model")
run in the latent space of the churches LDM, with every reverse step sandwiched between a decode and an encode.

One step of DDNM-main/functions/svd_ddnm.py::ddnm_diffusion, moved to latent space (z_t is the latent at step t):
    eps    = UNet(z_t, t)
    z0_t   = (z_t - sqrt(1 - a_t) * eps) / sqrt(a_t)          # Eq. 12   (latent x0 prediction)
    x0_t   = D(z0_t)                                          # decode   -> pixel space
    x0_hat = x0_t - A^+ (A x0_t - y)                          # Eq. 17   (range/null-space projection, pixel space)
    z0_hat = E(x0_hat)                                        # encode   -> latent space
    z_next = sqrt(a_next) z0_hat + c1 * noise + c2 * eps      # DDIM step, c1 = sqrt(1-a_next) eta, c2 = sqrt(1-a_next) sqrt(1-eta^2)
D / E come from a codec: the SLD VAE (with the latent scaling used in training), the SPNN, or VAE-encode + SPNN-decode.
The time schedule (with optional RePaint-style time travel) and alpha helper are DDNM's own.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torchvision.utils import make_grid
from torchvision.transforms.functional import to_pil_image

ROOT = Path(__file__).resolve().parent.parent.parent            # .../churches-reboot
_ddnm = ROOT / "DDNM-main"
if str(_ddnm) not in sys.path:
    sys.path.insert(0, str(_ddnm))
from functions.svd_ddnm import get_schedule_jump, compute_alpha   # noqa: E402  (DDNM-main)


# ----------------------------------------------------------------------------- codecs
class VAECodec:
    """SLD VAE encode/decode in the scaled latent space used by the LDM (mu/sigma from cache/latent_stats.json)."""

    def __init__(self, vae, stats):
        self.vae, self.stats = vae, stats

    @torch.no_grad()
    def encode(self, x):
        moments = self.vae.quant_conv(self.vae.encoder(x)).float()
        mean = torch.chunk(moments, 2, dim=1)[0]
        return self.stats.to_scaled(mean)

    @torch.no_grad()
    def decode(self, z):
        return torch.nan_to_num(self.vae.decode(self.stats.from_scaled(z)).float())


class SPNNCodec:
    """SPNN encode/decode; its latent space is the same scaled space (it was trained on those latents)."""

    def __init__(self, spnn):
        self.spnn = spnn

    @torch.no_grad()
    def encode(self, x):
        return self.spnn.encode(x).float()

    @torch.no_grad()
    def decode(self, z):
        return torch.nan_to_num(self.spnn.decode(z).float())


class VAEEncSPNNDecCodec:
    """Hybrid codec: encode with the SLD VAE (scaled latent), decode with the SPNN decoder."""

    def __init__(self, vae, stats, spnn):
        self._enc, self._dec = VAECodec(vae, stats), SPNNCodec(spnn)

    def encode(self, x):
        return self._enc.encode(x)

    def decode(self, z):
        return self._dec.decode(z)


class SDVAECodec:
    """Stable Diffusion (diffusers AutoencoderKL) encode/decode in the scaled latent space z = scaling * E(x).mean."""

    def __init__(self, vae, scaling=0.18215):
        self.vae, self.scaling = vae, scaling

    @torch.no_grad()
    def encode(self, x):
        return self.vae.encode(x.to(self.vae.dtype)).latent_dist.mean.float() * self.scaling

    @torch.no_grad()
    def decode(self, z):
        return torch.nan_to_num(self.vae.decode((z / self.scaling).to(self.vae.dtype)).sample.float())


class HybridCodec:
    """encode with one codec, decode with another (e.g. SD-VAE encode + SPNN decode)."""

    def __init__(self, enc_codec, dec_codec):
        self.enc_codec, self.dec_codec = enc_codec, dec_codec

    def encode(self, x):
        return self.enc_codec.encode(x)

    def decode(self, z):
        return self.dec_codec.decode(z)


# ----------------------------------------------------------------------------- sampler
@torch.no_grad()
def latent_ddnm(eps_model, betas, codec, A, Ap, y, latent_shape, T_sampling=100, eta=0.85,
                travel_length=1, travel_repeat=1, generator=None, record=True, stop_frac=1.0, sandwich_frac=1.0, clip_x0=False, noise_rule="ddnm"):
    """
    eps_model(z, t): noise predictor on latents, t int64 [n].  betas: [T] tensor of the LDM's beta schedule.
    codec: object with encode(x)->z and decode(z)->x.  A / Ap: degradation and its pseudo-inverse on images.
    y: degraded observation [n,3,H,W] in [-1,1].  Returns dict(x=final image, z=final latent, frames=[(t, x0_hat), ...]).
    stop_frac:     stop after this fraction of the steps and return the x0_hat of that step (1.0 = run to the end).
    sandwich_frac: after this fraction of the steps, skip the encode (plain latent DDIM on z0_t); the projection is still
                   applied to every x0_hat, so the known pixels stay exact in the returned image (1.0 = sandwich every step).
    clip_x0:       clamp the decoded x0 to [-1, 1] before the projection / re-encoding (needed for the SD VAE, whose
                   out-of-range decodes blow up when re-encoded; DDNM itself does not clip).
    noise_rule:    "ddnm": DDNM's update  z = sqrt(a') z0 + sqrt(1-a') (eta * noise + sqrt(1-eta^2) * eps)   (svd_ddnm.py)
                   "ddim": DDIM-paper variance sigma = eta * sqrt((1-a')/(1-a) * (1-a/a')),  z = sqrt(a') z0 + sqrt(1-a'-sigma^2) eps + sigma * noise
                           (what diffusers' DDIMScheduler does; much less fresh noise per step).
    """
    device, n = y.device, y.shape[0]
    T = betas.shape[0]
    skip = T // T_sampling
    z = torch.randn((n, *latent_shape), device=device, generator=generator)
    times = get_schedule_jump(T_sampling, travel_length, travel_repeat)
    frames, x0_hat, z0_hat = [], None, None
    n_steps = sum(1 for a, b in zip(times[:-1], times[1:]) if b < a); k_step = 0
    for i, j in zip(times[:-1], times[1:]):
        i, j = i * skip, j * skip
        if j < 0:
            j = -1
        t = torch.full((n,), i, device=device, dtype=torch.long)
        t_next = torch.full((n,), j, device=device, dtype=torch.long)
        at_next = compute_alpha(betas, t_next)                       # a_{-1} = 1 (DDNM pads the schedule)
        if j < i:                                                    # normal reverse step
            at = compute_alpha(betas, t)
            eps = eps_model(z, t).float()
            z0_t = (z - eps * (1 - at).sqrt()) / at.sqrt()           # Eq. 12
            x0_t = codec.decode(z0_t)                                # decode
            if clip_x0:
                x0_t = x0_t.clamp(-1, 1)
            x0_hat = x0_t - Ap(A(x0_t) - y)                          # Eq. 17
            k_step += 1
            z0_hat = codec.encode(x0_hat) if k_step <= sandwich_frac * n_steps else z0_t   # encode (or plain DDIM in the tail)
            if noise_rule == "ddim":
                sigma = eta * ((1 - at_next) / (1 - at) * (1 - at / at_next)).clamp_min(0).sqrt()
                c1, c2 = sigma, (1 - at_next - sigma ** 2).clamp_min(0).sqrt()
            else:
                c1 = (1 - at_next).sqrt() * eta
                c2 = (1 - at_next).sqrt() * ((1 - eta ** 2) ** 0.5)
            z = at_next.sqrt() * z0_hat + c1 * torch.randn(z.shape, device=device, generator=generator) + c2 * eps
            if record:
                frames.append((i, x0_hat.clamp(-1, 1).cpu()))
            if k_step >= stop_frac * n_steps:
                break
        else:                                                        # time travel: re-noise the last corrected estimate
            z = at_next.sqrt() * z0_hat + torch.randn(z.shape, device=device, generator=generator) * (1 - at_next).sqrt()
    return {"x": x0_hat.clamp(-1, 1).cpu(), "z": z.cpu(), "frames": frames}


# ----------------------------------------------------------------------------- inpainting helpers
def load_mask(path, size, device):
    """DDNM-main/exp/inp_masks/mask.npy (1 = known pixel, 0 = hole), resized with nearest neighbour -> [1,1,size,size]."""
    import numpy as np
    m = torch.from_numpy(np.load(path)).float()[None, None]
    return torch.nn.functional.interpolate(m, size=(size, size), mode="nearest").to(device)


def inpainting_ops(mask):
    A = lambda x: x * mask
    return A, A                                                      # A^+ = A for a masking operator


def box_mask(size, hole, device):
    """Centered square hole: [1,1,size,size] with 1 = known pixel, 0 = hole."""
    m = torch.ones(1, 1, size, size, device=device)
    a = (size - hole) // 2
    m[:, :, a:a + hole, a:a + hole] = 0
    return m


def chunk_mask(size, missing_frac, device):
    """One big centered square hole covering `missing_frac` of the image area: [1,1,size,size], 1 = known, 0 = hole."""
    return box_mask(size, int(round(size * missing_frac ** 0.5)), device)


def random_mask(size, missing_frac, device, seed=0):
    """Random pixel dropout: [1,1,size,size] with 1 = known pixel, 0 = missing; `missing_frac` of the pixels are holes."""
    g = torch.Generator().manual_seed(seed)
    keep = (torch.rand(1, 1, size, size, generator=g) >= missing_frac).float()
    return keep.to(device)


def mean_upsample(x, scale):
    """DDNM-main/guided_diffusion/diffusion.py::MeanUpsample (each low-res pixel repeated scale x scale)."""
    n, c, h, w = x.shape
    return (torch.zeros(n, c, h, scale, w, scale, device=x.device, dtype=x.dtype) + x.view(n, c, h, 1, w, 1)).view(n, c, h * scale, w * scale)


def color_ops():
    """DDNM's colorization: A = channel mean broadcast to 3 gray channels (color2gray), A^+ = identity (gray2color)."""
    A = lambda x: x.mean(1, keepdim=True).expand(-1, 3, -1, -1)
    Ap = lambda y: y
    return A, Ap


def sr_ops(scale, size):
    """DDNM's `sr_averagepooling`: A = average pooling by `scale`, A^+ = mean (nearest) upsampling."""
    pool = torch.nn.AdaptiveAvgPool2d((size // scale, size // scale))
    A = lambda x: pool(x)
    Ap = lambda y: mean_upsample(y, scale)
    return A, Ap


def to_pil(imgs, nrow=None):
    """[N,3,H,W] in [-1,1] -> one PIL image (panels side by side)."""
    imgs = imgs.detach().float().cpu().clamp(-1, 1)
    return to_pil_image(make_grid((imgs + 1) / 2, nrow=nrow or imgs.shape[0], padding=2))


def save_gif(pil_frames, path, duration=60, hold_last=15):
    frames = list(pil_frames) + [pil_frames[-1]] * hold_last
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=duration, loop=0)
    return path
