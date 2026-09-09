"""
Drop-in replacement of a diffusers AutoencoderKL by the SPNN-512 codec.

The SPNN checkpoint we evaluate is `imagenet_latent_ddnm/runs/spnn512_sd15_distill/
ckpt_last.pt` — SPNNAutoencoder512 distilled against SD 1.5's KL-VAE, so its latent
lives in the same 4x64x64 space the SD 1.5 UNet expects and it can be swapped into any
SD1.5-family pipeline by assigning `pipe.vae = build_spnn_vae(...)`.

Latent scale
------------
`imagenet_latent_ddnm/DDNM-main-scratch/configs/sd15_inpaint_latent.yml` leaves
`spnn_external_scale` unset with the note "SPNN-512 is prescaled into SD's latent
space", i.e. spnn.encode(x) already carries SD's 0.18215 factor. Every diffusers
pipeline, however, applies that factor itself -- multiplying after encode, dividing
before decode. So in the `prescaled` regime this shim must divide it back out on
encode and re-apply it on decode, or the factor lands twice in each direction:
the UNet would get latents 5.5x too SMALL (z*sf instead of z), and spnn.decode would
get latents 5.5x too LARGE (z/sf instead of z).

Getting this backwards silently poisons every number in the table, so
`verify_codec.py` measures which regime the checkpoint is actually in rather than
trusting the comment. Pass the measured answer as `latent_scale`.
"""

import os
import sys

import torch
import torch.nn as nn

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

SD15_SCALING_FACTOR = 0.18215


def load_spnn512(checkpoint: str, weights: str = "ema", device="cuda",
                 dtype=torch.float32, mix_type="householder", hidden=128,
                 r_hidden=256, scale_bound=1.0):
    """Instantiate SPNNAutoencoder512 and load `ckpt_last.pt`.

    The checkpoint stores two state dicts, `ema` and `model`, both with every key
    prefixed by `spnn.` (the training wrapper also held a discriminator). Config
    values default to what sd15_inpaint_latent.yml declares as "MUST match training".
    """
    from imagenet_latent_ddnm.spnn_model import SPNNAutoencoder512

    model = SPNNAutoencoder512(mix_type=mix_type, hidden=hidden,
                               r_hidden=r_hidden, scale_bound=scale_bound)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if weights not in state:
        avail = [k for k in state if isinstance(state[k], dict)]
        raise KeyError(f"'{weights}' not in checkpoint; available dicts: {avail}")
    sd = state[weights]
    stripped = {k[len("spnn."):]: v for k, v in sd.items() if k.startswith("spnn.")}
    if not stripped:
        stripped = sd
    missing, unexpected = model.load_state_dict(stripped, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"SPNN load mismatch — missing={len(missing)} unexpected={len(unexpected)}; "
            f"first missing={list(missing)[:5]} first unexpected={list(unexpected)[:5]}"
        )
    return model.to(device=device, dtype=dtype).eval().requires_grad_(False)


class _Dist:
    """Stands in for DiagonalGaussianDistribution; SPNN is deterministic."""
    def __init__(self, z):
        self.latents = z
        self.mean = z
    def sample(self, generator=None):
        return self.latents
    def mode(self):
        return self.latents


class _EncOut:
    def __init__(self, z):
        self.latent_dist = _Dist(z)
        self.latents = z
    def __getitem__(self, i):
        return (self.latent_dist,)[i]


class _DecOut:
    def __init__(self, x):
        self.sample = x
    def __getitem__(self, i):
        return (self.sample,)[i]


class SPNNVAEShim(nn.Module):
    """Presents SPNN-512 through the slice of the AutoencoderKL API diffusers uses.

    `config` is borrowed wholesale from the real VAE so that everything downstream —
    `scaling_factor`, `latent_channels`, `block_out_channels` (which sets the
    pipeline's vae_scale_factor of 8) — keeps reporting SD 1.5's values.
    """

    def __init__(self, spnn, vae_config, latent_scale="prescaled",
                 scaling_factor=SD15_SCALING_FACTOR):
        super().__init__()
        self.spnn = spnn
        self.config = vae_config
        self.sf = scaling_factor
        if latent_scale not in ("prescaled", "raw"):
            raise ValueError("latent_scale must be 'prescaled' or 'raw'")
        self.latent_scale = latent_scale

    @property
    def device(self):
        return next(self.spnn.parameters()).device

    @property
    def dtype(self):
        return next(self.spnn.parameters()).dtype

    def _to_unscaled(self, z):
        """SPNN encoder output -> the 'raw VAE' convention diffusers expects."""
        return z / self.sf if self.latent_scale == "prescaled" else z

    def _from_unscaled(self, z):
        return z * self.sf if self.latent_scale == "prescaled" else z

    @torch.no_grad()
    def encode(self, x, return_dict=True):
        z = self._to_unscaled(self.spnn.encode(x.to(self.dtype)))
        return _EncOut(z) if return_dict else (_Dist(z),)

    @torch.no_grad()
    def decode(self, z, return_dict=True, generator=None, **kwargs):
        x = self.spnn.decode(self._from_unscaled(z.to(self.dtype)))
        return _DecOut(x) if return_dict else (x,)

    # Pipelines call these for memory tuning; SPNN needs no special handling.
    def enable_slicing(self): pass
    def disable_slicing(self): pass
    def enable_tiling(self, *a, **k): pass
    def disable_tiling(self): pass


def build_spnn_vae(pipe, checkpoint, weights="ema", latent_scale="prescaled",
                   device="cuda", dtype=torch.float32, strict_scale=True,
                   **spnn_kwargs):
    """Swap `pipe.vae` for the SPNN codec, reusing the pipeline's own VAE config.

    Note the shim divides by `scaling_factor` on encode while the pipeline multiplies
    by it, so the two cancel and the UNet always receives `spnn.encode(x)` at SPNN's
    native scale. That is correct precisely because SPNN was distilled against SD 1.5's
    VAE and every editing model here is SD1.5-family, so every UNet expects latents at
    SD 1.5's 0.18215 scale.

    If some pipeline ever carried a different scaling factor, that cancellation would
    silently hand its UNet latents at the wrong scale instead of failing. So check it
    rather than let it pass quietly.
    """
    sf = float(pipe.vae.config.scaling_factor)
    if abs(sf - SD15_SCALING_FACTOR) > 1e-6:
        msg = (f"pipeline VAE scaling_factor is {sf}, not SD 1.5's "
               f"{SD15_SCALING_FACTOR}. SPNN-512 emits SD1.5-scale latents, so this "
               f"UNet would receive latents off by {sf / SD15_SCALING_FACTOR:.3f}x. "
               f"Pass strict_scale=False only if you have confirmed this is intended.")
        if strict_scale:
            raise ValueError(msg)
        print(f"  WARNING: {msg}")

    spnn = load_spnn512(checkpoint, weights=weights, device=device, dtype=dtype,
                        **spnn_kwargs)
    return SPNNVAEShim(spnn, pipe.vae.config, latent_scale=latent_scale,
                       scaling_factor=sf)
