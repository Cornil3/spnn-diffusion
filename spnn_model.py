import types
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# SD 1.x KL-VAE (diffusers): latents passed to the UNet use this scale after encode.
SD15_VAE_SCALING_FACTOR = 0.18215


class BaseOrthogonal1x1Conv(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels

    def _compute_W(self, device, dtype):
        raise NotImplementedError

    def forward(self, x):
        B, C, H, W = x.shape
        Wm = self._compute_W(x.device, x.dtype).view(C, C, 1, 1)
        return F.conv2d(x, Wm)

    def inverse(self, x):
        B, C, H, W = x.shape
        Wm = self._compute_W(x.device, x.dtype).t().view(C, C, 1, 1)
        return F.conv2d(x, Wm)


class Cayley1x1Conv(BaseOrthogonal1x1Conv):
    def __init__(self, channels, eps=1e-6):
        super().__init__(channels)
        self.eps = eps
        self.A_unconstrained = nn.Parameter(torch.zeros(channels, channels))

    def _compute_W(self, device, dtype):
        C = self.channels
        B = self.A_unconstrained.to(device=device, dtype=torch.float32)
        A = B - B.t()
        I = torch.eye(C, device=device, dtype=torch.float32)
        W = torch.linalg.solve(I + A + self.eps * I, I - A)
        return W.to(dtype=dtype)


class Householder1x1Conv(BaseOrthogonal1x1Conv):
    def __init__(self, channels, num_reflections=8, eps=1e-8):
        super().__init__(channels)
        self.num_reflections = num_reflections
        self.eps = eps
        if num_reflections > 0:
            self.V = nn.Parameter(torch.randn(num_reflections, channels))
        else:
            self.register_parameter("V", None)

    def _compute_W(self, device, dtype):
        C = self.channels
        if self.V is None or self.num_reflections == 0:
            return torch.eye(C, device=device, dtype=dtype)
        W = torch.eye(C, device=device, dtype=dtype)
        V = self.V.to(device=device, dtype=dtype)
        for i in range(self.num_reflections):
            v = V[i]
            v = v / (v.norm(p=2) + self.eps)
            H = torch.eye(C, device=device, dtype=dtype) - 2.0 * torch.outer(v, v)
            W = H @ W
        return W


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(min(32, channels), channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(min(32, channels), channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.block(x) + x)


class SelfAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(min(32, channels), channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.out = nn.Conv2d(channels, channels, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, 3, C, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        attn = torch.bmm(q.transpose(1, 2), k) * (C ** -0.5)
        attn = attn.softmax(dim=-1)
        out = torch.bmm(v, attn.transpose(1, 2)).reshape(B, C, H, W)
        return x + self.out(out)


class ConvMLP(nn.Module):
    def __init__(self, in_ch, out_ch, scale_bound, hidden_ch=128, feat_size=None):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.scale_bound = scale_bound

        if in_ch == 0:
            self.net = nn.Parameter(torch.zeros(1, out_ch, 1, 1))
        elif feat_size is not None and feat_size == 1:
            h = min(max(hidden_ch, in_ch), 512)
            self.net = nn.Sequential(
                nn.Conv2d(in_ch, h, 1), nn.ReLU(),
                nn.Conv2d(h, h, 1), nn.ReLU(),
                nn.Conv2d(h, out_ch, 1),
            )
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)
        elif feat_size is not None and feat_size >= 4:
            self._use_residual = True
            h1 = hidden_ch
            h2 = hidden_ch * 2
            self.enc_in = nn.Sequential(
                nn.Conv2d(in_ch, h1, 3, padding=1),
                nn.GroupNorm(min(32, h1), h1),
                nn.ReLU(inplace=True),
            )
            self.enc_blocks = nn.Sequential(
                ResBlock(h1), ResBlock(h1), ResBlock(h1),
            )
            self.down = nn.Sequential(
                nn.Conv2d(h1, h2, 3, stride=2, padding=1),
                nn.GroupNorm(min(32, h2), h2),
                nn.ReLU(inplace=True),
            )
            self.bottleneck = nn.Sequential(
                ResBlock(h2), ResBlock(h2),
                SelfAttention(h2),
            )
            self.up = nn.Sequential(
                nn.ConvTranspose2d(h2, h1, 4, stride=2, padding=1),
                nn.GroupNorm(min(32, h1), h1),
            )
            self.dec_blocks = nn.Sequential(
                ResBlock(h1), ResBlock(h1), ResBlock(h1),
            )
            self.out = nn.Conv2d(h1, out_ch, 3, padding=1)
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)
        else:
            h = min(max(hidden_ch, in_ch), 512)
            self.net = nn.Sequential(
                nn.Conv2d(in_ch, h, 3, padding=1), nn.ReLU(),
                nn.Conv2d(h, h, 3, padding=1), nn.ReLU(),
                nn.Conv2d(h, out_ch, 3, padding=1),
            )
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

    def forward(self, x, neg=False):
        if self.in_ch > 0:
            if getattr(self, '_use_residual', False):
                h = self.enc_in(x)
                h = self.enc_blocks(h)
                skip = h
                h = self.down(h)
                h = self.bottleneck(h)
                h = F.relu(self.up(h) + skip)
                h = self.dec_blocks(h)
                x = self.out(h)
            else:
                x = self.net(x)
        else:
            B, _, H, W = x.shape
            x = self.net.expand(B, self.out_ch, H, W)

        if self.scale_bound is not None:
            x = torch.tanh(x) * self.scale_bound
            if neg:
                x = -x
            x = x.exp()
        else:
            pass
            x = torch.tanh(x)
        return x


class PixelUnshuffleBlock(nn.Module):
    def __init__(self, r: int):
        super().__init__()
        self.r = r

    def forward(self, x):
        return F.pixel_unshuffle(x, self.r)

    def pinv(self, y):
        return F.pixel_shuffle(y, self.r)


class ConvPINNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, hidden=128, r_hidden=None, scale_bound=2.0,
                 mix_type="cayley", feat_size=None):
        super().__init__()
        assert in_ch > out_ch, (
            f"ConvPINNBlock requires in_ch > out_ch (got {in_ch}, {out_ch})"
        )
        self.in_ch = in_ch
        self.out_ch = out_ch
        if r_hidden is None:
            r_hidden = hidden * 2

        side_ch = in_ch - out_ch
        self.t = ConvMLP(side_ch, out_ch, None, hidden, feat_size=feat_size)
        self.s = ConvMLP(side_ch, out_ch, scale_bound, hidden, feat_size=feat_size)
        self.r = ConvMLP(out_ch, side_ch, None, r_hidden, feat_size=feat_size)

        if mix_type == "householder":
            self.mix = Householder1x1Conv(in_ch)
        else:
            self.mix = Cayley1x1Conv(in_ch)

    def forward(self, x):
        x = self.mix.forward(x)
        x0 = x[:, :self.out_ch]
        x1 = x[:, self.out_ch:]
        y = x0 * self.s(x1) + self.t(x1)
        return y

    def pinv(self, y):
        x1 = self.r(y)
        x0 = (y - self.t(x1)) * self.s(x1, neg=True)
        x = torch.cat([x0, x1], dim=1)
        return self.mix.inverse(x)


# ─── Original architectures (kept for backward compat) ───

class SPNNAutoencoder(nn.Module):
    def __init__(self, mix_type="cayley", hidden=96, r_hidden=192, scale_bound=2.0):
        super().__init__()
        self.blocks = nn.ModuleList([
            PixelUnshuffleBlock(2),
            ConvPINNBlock(12, 3, hidden=hidden, r_hidden=r_hidden,
                          scale_bound=scale_bound, mix_type=mix_type, feat_size=16),
        ])

    def forward(self, x):
        return self.encode(x)

    def encode(self, x):
        for b in self.blocks:
            x = b(x)
        return x

    def decode(self, y):
        for b in reversed(self.blocks):
            y = b.pinv(y)
        return y


class SPNNAutoencoder128(nn.Module):
    """SPNN autoencoder for 128x128 images. Output: 3x32x32 latent."""
    def __init__(self, mix_type="cayley", hidden=96, r_hidden=192, scale_bound=2.0):
        super().__init__()
        self.blocks = nn.ModuleList([
            PixelUnshuffleBlock(2),
            PixelUnshuffleBlock(2),
            ConvPINNBlock(48, 3, hidden=hidden, r_hidden=r_hidden,
                          scale_bound=scale_bound, mix_type=mix_type, feat_size=32),
        ])

    def forward(self, x):
        return self.encode(x)

    def encode(self, x):
        for b in self.blocks:
            x = b(x)
        return x

    def decode(self, y):
        for b in reversed(self.blocks):
            y = b.pinv(y)
        return y


# ─── New: 256×256 → 3×64×64 (matches CompVis LDM-CelebAHQ VQ-VAE) ───

class SPNNAutoencoder256(nn.Module):
    """
    SPNN autoencoder for 256×256 images → 3×64×64 latent.
    Matches the CompVis/ldm-celebahq-256 VQ-VAE latent shape (f=4, 3 channels).

    Architecture:
        3×256×256  →[PixelUnshuffle(4)]→  48×64×64
        48×64×64   →[ConvPINNBlock]→       3×64×64
    """
    def __init__(self, mix_type="cayley", hidden=192, r_hidden=384, scale_bound=2.0):
        super().__init__()
        # Single PixelUnshuffle(4): 3×256×256 → 3*16=48 × 64×64
        # Then ConvPINN compresses 48→3 channels at 64×64
        self.blocks = nn.ModuleList([
            PixelUnshuffleBlock(4),          # 3×256×256 → 48×64×64
            ConvPINNBlock(48, 3, hidden=hidden, r_hidden=r_hidden,
                          scale_bound=scale_bound, mix_type=mix_type, feat_size=64),
                                             # 48×64×64 → 3×64×64
        ])

    def forward(self, x):
        return self.encode(x)

    def encode(self, x):
        for b in self.blocks:
            x = b(x)
        return x

    def decode(self, y):
        for b in reversed(self.blocks):
            y = b.pinv(y)
        return y

class SPNNAutoencoder512(nn.Module):
    """
    SPNN autoencoder for 512×512 images → 4×64×64 latent.
    Matches the SD 1.5 KL-VAE latent shape (f=8, 4 channels).
    Architecture:
        3×512×512  →[PixelUnshuffle(4)]→  48×128×128
        48×128×128 →[ConvPINNBlock]→       16×128×128
        16×128×128 →[PixelUnshuffle(2)]→   64×64×64
        64×64×64   →[ConvPINNBlock]→       4×64×64
    """
    def __init__(self, mix_type="cayley", hidden=256, r_hidden=256, scale_bound=2.0):
        super().__init__()
        # Encoder output is in the same regime as SD KL-VAE *before* scaling; latent_ddnm and
        # diffusers-style helpers multiply/divide by this so the UNet sees SD-scaled latents.
        self.config = types.SimpleNamespace(scaling_factor=SD15_VAE_SCALING_FACTOR)
        self.blocks = nn.ModuleList([
            # Stage 1: 3x256x256 -> 48x64x64
            PixelUnshuffleBlock(4),
            # Stage 2: 48x64x64 -> 16x64x64
            ConvPINNBlock(48, 16, hidden=hidden, r_hidden=r_hidden,
                          scale_bound=scale_bound,
                          mix_type=mix_type, feat_size=64),
            # Stage 3: 16x64x64 -> 64x32x32
            PixelUnshuffleBlock(2),
            # Stage 4: 64x32x32 -> 4x32x32  (latent)
            ConvPINNBlock(64, 4, hidden=hidden, r_hidden=r_hidden,
                          scale_bound=scale_bound,
                          mix_type=mix_type, feat_size=32),
        ])

    def forward(self, x):
        return self.encode(x)

    def encode(self, x):
        for b in self.blocks:
            x = b(x)
        return x

    def decode(self, y):
        for b in reversed(self.blocks):
            y = b.pinv(y)
        return y



# ─── Wrapper classes for pipeline integration ───

class SPNNLatentWrapper(nn.Module):
    """Wraps SPNN for use with LatentDiffusionModel."""
    def __init__(self, spnn):
        super().__init__()
        self.spnn = spnn
        self.embed_dim = 3
        self._decoder = type('Decoder', (), {'z_shape': (1, 3, 16, 16)})()

    @property
    def decoder(self):
        return self._decoder

    def encode(self, x):
        # LatentDiffusion.get_first_stage_encoding expects a Tensor or DiagonalGaussianDistribution.
        return self.spnn.encode(x)

    def decode(self, z):
        return self.spnn.decode(z)


class _SPNNDummyQuantizer(nn.Module):
    """VQ-compatible no-op so LatentDiffusion sampling paths that call `quantize` still work."""

    def forward(self, h):
        return h, h.new_zeros(()), (0, 0, torch.zeros(h.shape[0], dtype=torch.long, device=h.device))


class SPNNFirstStage256(nn.Module):
    """
    First-stage model for LatentDiffusion: SPNN 256→64²×3 latent (same shape as CelebAHQ VQ-f4).
    Load weights with `ckpt_path` (e.g. best.pt from SPNNAutoencoder256 training).
    """

    def __init__(self, ckpt_path=None):
        super().__init__()
        self.spnn = SPNNAutoencoder256()
        self.embed_dim = 3
        self._decoder = type('Decoder', (), {'z_shape': (1, 3, 64, 64)})()
        self.quantize = _SPNNDummyQuantizer()
        if ckpt_path is not None:
            self._load_ckpt(ckpt_path)

    @property
    def decoder(self):
        return self._decoder

    def _load_ckpt(self, path):
        ck = torch.load(path, map_location='cpu')
        if isinstance(ck, dict):
            if 'state_dict' in ck:
                sd = ck['state_dict']
            elif 'model' in ck:
                sd = ck['model']
            else:
                sd = ck
        else:
            sd = ck
        if not isinstance(sd, dict):
            raise ValueError(f'Unexpected checkpoint type from {path}')
        new_sd = {}
        for k, v in sd.items():
            nk = k
            for p in ('spnn.', 'module.', 'model.'):
                if nk.startswith(p):
                    nk = nk[len(p):]
            new_sd[nk] = v
        missing, unexpected = self.spnn.load_state_dict(new_sd, strict=False)
        print(f'Loaded SPNN weights from {path} (missing={len(missing)}, unexpected={len(unexpected)})')

    def encode(self, x):
        return self.spnn.encode(x)

    def decode(self, z, force_not_quantize=False):
        return self.spnn.decode(z)


class SPNNLatentWrapper128(SPNNLatentWrapper):
    """Wraps SPNNAutoencoder128 for use with LatentDiffusionModel."""
    pass


class SPNNLatentWrapper256(SPNNLatentWrapper):
    """Wraps SPNNAutoencoder256 for use with CompVis LDM-CelebAHQ pipeline."""
    def __init__(self, spnn):
        super().__init__(spnn)
        self._decoder = type('Decoder', (), {'z_shape': (1, 3, 64, 64)})()


class _SPNNLatentDist:
    def __init__(self, z):
        self._z = z

    def mode(self):
        return self._z


class _SPNNEncodeOutput:
    def __init__(self, z):
        self.latent_dist = _SPNNLatentDist(z)


class SPNN512VAEAdapter(nn.Module):
    """
    Diffusers-like VAE surface for latent_ddnm: encode() -> latent_dist.mode(),
    decode(z) -> object with .sample. Latent shape matches SD 1.5 (4×64×64 for 512² input).

    **Base vs inpainting UNet:** The finetuned inpainting model still uses the **same 4-channel
    latent tensor** as SD v1.5; the UNet simply **concatenates** extra inputs (1-channel latent
    mask + 4-channel masked-image latents). SPNN512 does **not** need different channel counts
    for inpainting — use the same adapter and the same ``scaling_factor`` as for the base model.

    **scaling_factor:** Should match what the diffusion UNet was trained with — for HuggingFace
    SD 1.5 / SD inpainting, that is ``pipe.vae.config.scaling_factor`` (typically
    :data:`SD15_VAE_SCALING_FACTOR`). Set ``scaling_factor=1.0`` only if your SPNN was trained so
    its encoder output already lives in the **same numeric space** as the UNet latents without
    this extra scale (uncommon if you matched SD KL latents explicitly).
    """

    def __init__(self, spnn: SPNNAutoencoder512, scaling_factor: float = SD15_VAE_SCALING_FACTOR):
        super().__init__()
        self.spnn = spnn
        self.config = types.SimpleNamespace(scaling_factor=float(scaling_factor))

    def encode(self, x):
        z = self.spnn.encode(x)
        return _SPNNEncodeOutput(z)

    def decode(self, z):
        x = self.spnn.decode(z)
        return types.SimpleNamespace(sample=x)


# Default kwargs for SPNNAutoencoder256 (matches common training / your load_spnn() snippet)
SPNN256_DEFAULT_KWARGS = dict(mix_type="cayley", hidden=192, r_hidden=384, scale_bound=2.0)


class SPNN256VAEAdapter(nn.Module):
    """
    SPNNAutoencoder256: 256² RGB → 3×64×64 latent. SD inpainting expects 4×64×64 — we **pad** a zero 4th
    channel for the UNet; decode uses only the first 3 channels. Images are resized 512↔256 inside
    encode/decode when `decode_size` is set (e.g. 512 for the latent-DDNM notebooks).
    """

    def __init__(self, spnn: SPNNAutoencoder256, decode_size: Optional[int] = 512):
        super().__init__()
        self.spnn = spnn
        self.decode_size = decode_size
        self.config = types.SimpleNamespace(scaling_factor=1.0)

    def encode(self, x):
        if x.shape[-1] != 256 or x.shape[-2] != 256:
            x = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False)
        z3 = self.spnn.encode(x)
        b, _, h, w = z3.shape
        pad = z3.new_zeros(b, 1, h, w)
        z4 = torch.cat([z3, pad], dim=1)
        return _SPNNEncodeOutput(z4)

    def decode(self, z):
        z3 = z[:, :3]
        x = self.spnn.decode(z3)
        if self.decode_size is not None and (x.shape[-1] != self.decode_size or x.shape[-2] != self.decode_size):
            x = F.interpolate(
                x, size=(self.decode_size, self.decode_size), mode="bilinear", align_corners=False
            )
        return types.SimpleNamespace(sample=x)


def load_spnn256_vae_adapter(
    ckpt_path,
    device="cpu",
    dtype=torch.float32,
    decode_size: Optional[int] = 512,
    **spnn_kwargs,
):
    """
    Load **SPNNAutoencoder256** the same way as a typical training checkpoint:

        spnn = SPNNAutoencoder256(**SPNN_KWARGS).to(device).eval()
        sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        spnn.load_state_dict(sd)

    Merges **spnn_kwargs** with :data:`SPNN256_DEFAULT_KWARGS`. Tries ``strict=True`` first; on
    failure, strips ``module.`` / ``spnn.`` / … prefixes and loads with ``strict=False``.
    """
    import os

    path = os.path.expanduser(ckpt_path)
    kw = {**SPNN256_DEFAULT_KWARGS, **spnn_kwargs}
    spnn = SPNNAutoencoder256(**kw)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    if not isinstance(sd, dict):
        raise ValueError(f"Expected state dict in {path}")

    try:
        spnn.load_state_dict(sd, strict=True)
        print(f"SPNN256: loaded {path} (strict=True, exact key match)")
    except Exception:
        new_sd = {}
        for k, v in sd.items():
            nk = k
            for p in ("spnn.", "module.", "model.", "ae.", "net."):
                if nk.startswith(p):
                    nk = nk[len(p) :]
            new_sd[nk] = v
        missing, unexpected = spnn.load_state_dict(new_sd, strict=False)
        print(f"SPNN256: loaded {path} with key remap — missing={len(missing)}, unexpected={len(unexpected)}")
        if len(missing) > 20:
            print(
                "WARNING: many SPNN256 params missing — checkpoint may be wrong architecture or run."
            )

    spnn = spnn.to(device=device, dtype=dtype)
    spnn.eval()
    for p in spnn.parameters():
        p.requires_grad_(False)
    return SPNN256VAEAdapter(spnn, decode_size=decode_size)


def load_spnn512_vae_adapter(
    ckpt_path,
    device="cpu",
    dtype=torch.float32,
    verbose=True,
    scaling_factor: Optional[float] = None,
):
    """Load SPNNAutoencoder512 weights from best.pt (or similar) and return SPNN512VAEAdapter.

    **Checkpoint must match SPNNAutoencoder512** (512² → 4×64×64 latent). A checkpoint trained for
    `SPNNAutoencoder256` or another architecture will show hundreds of *missing* keys — the model
    then stays mostly **random** (strict=False only loads matching tensors).

    **scaling_factor:** Pass ``float(pipe.vae.config.scaling_factor)`` so SPNN latents match the
    SD / SD-inpainting UNet (default :data:`SD15_VAE_SCALING_FACTOR` when ``None``).
    """
    import os

    path = os.path.expanduser(ckpt_path)
    spnn = SPNNAutoencoder512()
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ck, dict):
        if "state_dict" in ck:
            sd = ck["state_dict"]
        elif "model" in ck:
            sd = ck["model"]
        else:
            sd = ck
    else:
        sd = ck
    if not isinstance(sd, dict):
        raise ValueError(f"Unexpected checkpoint type from {path}")
    new_sd = {}
    for k, v in sd.items():
        nk = k
        for p in ("spnn.", "module.", "model.", "ae.", "net."):
            if nk.startswith(p):
                nk = nk[len(p) :]
        new_sd[nk] = v
    missing, unexpected = spnn.load_state_dict(new_sd, strict=False)
    n_m, n_u = len(missing), len(unexpected)
    print(f"SPNN512 load_state_dict: missing={n_m}, unexpected={n_u}")
    if n_m > 20:
        print(
            "WARNING: Most SPNN512 weights did not load. Results use a near-random autoencoder.\n"
            "  → Use a checkpoint trained/saved for SPNNAutoencoder512 (same blocks as in spnn_model.py).\n"
            "  → Checkpoints for SPNNAutoencoder256 or full SD pipelines will NOT match."
        )
        if verbose:
            mk = list(missing)[:8]
            uk = list(unexpected)[:8]
            sk = list(new_sd.keys())[:8]
            print(f"  Sample keys NOT loaded (still random): {mk}")
            print(f"  Sample keys FROM checkpoint (after strip): {sk}")
            print(f"  Checkpoint keys with no model param: {uk}")
    spnn = spnn.to(device=device, dtype=dtype)
    spnn.eval()
    sf = SD15_VAE_SCALING_FACTOR if scaling_factor is None else float(scaling_factor)
    return SPNN512VAEAdapter(spnn, scaling_factor=sf)