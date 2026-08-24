"""
Glue that lets the ORIGINAL PSLD code run against SD1.5 + a swappable codec.

Nothing here reimplements PSLD. The sampler is `ldm/models/diffusion/psld.py` from the
attached PSLD repo, used unmodified, and the degradations are DPS's own operators from
`diffusion-posterior-sampling/guided_diffusion/measurements.py`. This module only supplies:

  * add_psld_paths() - puts both repos on sys.path (and stubs the one missing submodule)
  * SD15Shim        - the handful of LatentDiffusion attributes psld.py actually touches,
                      backed by the diffusers SD1.5 UNet and a codec of your choice
  * GradCodec       - latent_codec.py's codecs with autograd left on (PSLD differentiates
                      the measurement and gluing losses through the decoder)
  * colorization_operator() - the one degradation DPS does not ship, in DPS's own API

The codec swap happens in exactly one place: `differentiable_decode_first_stage` /
`encode_first_stage` below. That is the whole experiment -- PSLD's gluing term is
`|| z0 - E(A^T y + (I - A^T A) D(z0)) ||`, which the paper introduces (Sec. 2.1) to "guide
the diffusion process to sample latents for which the decoding-encoding map is not lossy".
SPNN's E(D(.)) is exactly the identity by construction, so that term costs it nothing.
"""

import os
import sys
import types

import torch

PSLD_ROOT = "/home/ron.libman/PSLD/PSLD-main"


def add_psld_paths(root=PSLD_ROOT):
    """sys.path for the attached PSLD repo: the LDM sampler and DPS's operators.

    The archive nests DPS's repo root at diffusion-posterior-sampling/util/, so both that
    and its parent are needed. util/motionblur/ ships empty (it is a git submodule), and
    measurements.py imports it at module level, so stub it -- it is only used by the
    motion_blur operator, which nothing here calls.
    """
    dps = os.path.join(root, "diffusion-posterior-sampling")
    for p in (os.path.join(root, "stable-diffusion"), dps, os.path.join(dps, "util")):
        if p not in sys.path:
            sys.path.insert(0, p)

    if "motionblur" not in sys.modules:
        pkg, sub = types.ModuleType("motionblur"), types.ModuleType("motionblur.motionblur")
        sub.Kernel, pkg.motionblur = None, sub
        sys.modules["motionblur"], sys.modules["motionblur.motionblur"] = pkg, sub


class GradCodec:
    """latent_codec.py's VAECodec / SPNNCodec with the graph left intact.

    Those classes decorate encode/decode with @torch.no_grad, which is right for DDNM (it
    only needs values) but silently kills PSLD, whose whole update is a gradient through
    the decoder. Codec weights stay frozen; only d/d(input) is ever needed.
    """

    def __init__(self, codec, name=None):
        self.codec = codec
        self.name = name or type(codec).__name__
        if hasattr(codec, "vae"):
            sf = codec.sf
            self._enc = lambda x: codec.vae.encode(x).latent_dist.mode() * sf
            self._dec = lambda z: codec.vae.decode(z / sf).sample
        elif hasattr(codec, "spnn"):
            sf = codec.sf                       # None => SPNN is prescaled into SD's latent space
            self._enc = lambda x: codec.spnn.encode(x) if sf is None else codec.spnn.encode(x) * sf
            self._dec = lambda z: codec.spnn.decode(z if sf is None else z / sf)
        else:
            raise TypeError(f"expected a codec with .vae or .spnn, got {type(codec).__name__}")

    def encode(self, x): return self._enc(x)
    def decode(self, z): return self._dec(z)


class SD15Shim:
    """Minimal stand-in for ldm's LatentDiffusion, so psld.DDIMSampler runs unmodified.

    psld.py touches exactly these: num_timesteps, alphas_cumprod, alphas_cumprod_prev,
    betas, device, parameterization, apply_model, differentiable_decode_first_stage,
    encode_first_stage, get_first_stage_encoding (and first_stage_model.quantize, only
    under quantize_denoised=True, which we never set).
    """

    def __init__(self, unet, codec, alphas_cumprod, context, device="cuda"):
        self.unet = unet
        self.codec = codec if isinstance(codec, GradCodec) else GradCodec(codec)
        self.context = context
        self.device = torch.device(device)

        self.alphas_cumprod = alphas_cumprod
        self.alphas_cumprod_prev = torch.cat([alphas_cumprod.new_ones(1), alphas_cumprod[:-1]])
        self.betas = 1.0 - alphas_cumprod / self.alphas_cumprod_prev   # exact inverse of the cumprod
        self.num_timesteps = alphas_cumprod.numel()
        self.parameterization = "eps"
        self.first_stage_model = None

    def apply_model(self, z, t, c=None):
        """eps_theta(z_t, t, c). SD1.5's UNet is conditional; PSLD passes the empty prompt."""
        ctx = self.context if c is None else c
        if ctx.shape[0] != z.shape[0]:
            ctx = ctx.expand(z.shape[0], -1, -1)
        return self.unet(z, t, encoder_hidden_states=ctx).sample

    # --- first stage: THE codec swap ---------------------------------------------------
    def differentiable_decode_first_stage(self, z):
        return self.codec.decode(z)                      # graph kept: PSLD backprops here

    def encode_first_stage(self, x):
        return self.codec.encode(x)                      # already in SD's *scaled* latent space

    def get_first_stage_encoding(self, z):
        return z                                         # ...so the usual rescale is a no-op

    @torch.no_grad()
    def decode_first_stage(self, z):
        return self.codec.decode(z)


def colorization_operator():
    """Grayscale measurement, in DPS's LinearOperator API (DPS ships no colorization op).

    A averages the 3 channels and keeps the result as 3 equal channels, so A is an
    orthogonal projector and A^T = A^+ = A on its range -- the same operator DDNM uses.
    """
    from guided_diffusion.measurements import LinearOperator

    class ColorizationOperator(LinearOperator):
        def forward(self, data, **kwargs):
            return data.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)

        def transpose(self, data, **kwargs):
            return data[:, :1].repeat(1, 3, 1, 1)

        def ortho_project(self, data, **kwargs):
            return data - self.transpose(self.forward(data))

        def project(self, data, measurement, **kwargs):
            return self.ortho_project(data) + measurement

    return ColorizationOperator()
