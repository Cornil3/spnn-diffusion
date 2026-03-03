import torch
import torch.nn as nn
import torch.nn.functional as F


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
        B = self.A_unconstrained.to(device=device, dtype=dtype)
        A = B - B.t()
        I = torch.eye(C, device=device, dtype=dtype)
        W = torch.linalg.solve(I + A + self.eps * I, I - A)
        return W


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


class ConvMLP(nn.Module):
    """
    Generic convolutional MLP used for s, t, r sub-networks.
    Builds a small U-Net-like arch for spatial features, or pointwise for 1x1.
    """
    def __init__(self, in_ch, out_ch, scale_bound, hidden_ch=128, feat_size=None):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.scale_bound = scale_bound

        if in_ch == 0:
            # Learned constant bias
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
            h1 = min(max(hidden_ch, in_ch), 512)
            h2 = min(h1 * 2, 1024)
            self.net = nn.Sequential(
                nn.Conv2d(in_ch, h1, 3, padding=1), nn.ReLU(),
                nn.Conv2d(h1, h1, 3, padding=1), nn.ReLU(),
                nn.Conv2d(h1, h2, 3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(h2, h2, 3, padding=1), nn.ReLU(),
                nn.ConvTranspose2d(h2, h1, 4, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(h1, out_ch, 3, padding=1),
            )
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)
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
            x = torch.tanh(x)
        return x


class PixelUnshuffleBlock(nn.Module):
    def __init__(self, r: int):
        super().__init__()
        self.r = r

    def forward(self, x, return_latent=False):
        return F.pixel_unshuffle(x, self.r), None

    def pinv(self, y, x1_override=None):
        return F.pixel_shuffle(y, self.r)


class ConvPINNBlock(nn.Module):
    """
    Coupling block: splits channels, applies affine transform via s,t,
    reconstructs via r (pseudo-inverse path).
    """
    def __init__(self, in_ch, out_ch, hidden=128, scale_bound=2.0,
                 mix_type="cayley", feat_size=None):
        super().__init__()
        assert in_ch > out_ch, (
            f"ConvPINNBlock requires in_ch > out_ch (got {in_ch}, {out_ch})"
        )
        self.in_ch = in_ch
        self.out_ch = out_ch

        side_ch = in_ch - out_ch
        self.t = ConvMLP(side_ch, out_ch, None, hidden, feat_size=feat_size)
        self.s = ConvMLP(side_ch, out_ch, scale_bound, hidden, feat_size=feat_size)
        self.r = ConvMLP(out_ch, side_ch, None, hidden, feat_size=feat_size)

        if mix_type == "householder":
            self.mix = Householder1x1Conv(in_ch)
        else:
            self.mix = Cayley1x1Conv(in_ch)

    def forward(self, x, return_latent=False):
        x = self.mix.forward(x)
        x0 = x[:, :self.out_ch]
        x1 = x[:, self.out_ch:]
        y = x0 * self.s(x1) + self.t(x1)
        z = x1 if return_latent else None
        return y, z

    def pinv(self, y, x1_override=None):
        x1 = self.r(y) if x1_override is None else x1_override
        x0 = (y - self.t(x1)) * self.s(x1, neg=True)
        x = torch.cat([x0, x1], dim=1)
        return self.mix.inverse(x)


class SPNNAutoencoder(nn.Module):
    """
    SPNN-based autoencoder for 256x256 images.

    Encoder path (forward):
        3x256x256  ->  pixel_unshuffle(4)  ->  48x64x64
        48x64x64   ->  ConvPINN(48->16)    ->  16x64x64
        16x64x64   ->  pixel_unshuffle(2)  ->  64x32x32
        64x32x32   ->  ConvPINN(64->4)     ->  4x32x32   (latent)

    Decoder path (pinv): reverses the above using s,t from forward + trained r networks.
    The latent is 4x32x32 = 4096 values, matching the SD-VAE latent shape.
    """
    def __init__(self, mix_type="cayley", hidden=128, scale_bound=2.0):
        super().__init__()
        self.blocks = nn.ModuleList([
            # Stage 1: 3x256x256 -> 48x64x64
            PixelUnshuffleBlock(4),
            # Stage 2: 48x64x64 -> 16x64x64
            ConvPINNBlock(48, 16, hidden=hidden, scale_bound=scale_bound,
                          mix_type=mix_type, feat_size=64),
            # Stage 3: 16x64x64 -> 64x32x32
            PixelUnshuffleBlock(2),
            # Stage 4: 64x32x32 -> 4x32x32  (latent)
            ConvPINNBlock(64, 4, hidden=hidden, scale_bound=scale_bound,
                          mix_type=mix_type, feat_size=32),
        ])

    def encode(self, x, return_latents=False):
        """Encode image -> latent. Optionally return side-info for perfect inversion."""
        latents = []
        for b in self.blocks:
            x, z = b(x, return_latent=return_latents)
            latents.append(z)
        return (x, latents) if return_latents else x

    def decode(self, y, latents=None):
        """Decode latent -> image using pseudo-inverse."""
        if latents is not None:
            z_stack = list(reversed(latents))
        else:
            z_stack = [None] * len(self.blocks)
        for b in reversed(self.blocks):
            z = z_stack.pop(0)
            y = b.pinv(y, x1_override=z)
        return y

    def forward(self, x, return_latents=False):
        """Full encode-decode cycle."""
        if return_latents:
            latent, side = self.encode(x, return_latents=True)
            recon = self.decode(latent, latents=side)
            return recon, latent, side
        else:
            latent = self.encode(x)
            recon = self.decode(latent)
            return recon, latent