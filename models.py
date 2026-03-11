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
        # Always compute in fp32 — fp16 makes (I+A) singular for large A
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
    """Conv3x3 → GroupNorm → ReLU → Conv3x3 → GroupNorm + skip → ReLU"""
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
    """Channel-wise self-attention: norm → qkv → attention → out + skip"""
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
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]  # each [B, C, HW]
        attn = torch.bmm(q.transpose(1, 2), k) * (C ** -0.5)  # [B, HW, HW]
        attn = attn.softmax(dim=-1)
        out = torch.bmm(v, attn.transpose(1, 2)).reshape(B, C, H, W)  # [B, C, H, W]
        return x + self.out(out)


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
            self._use_residual = True
            h1 = hidden_ch
            h2 = hidden_ch * 2
            # Encoder: project in + 3 residual blocks
            self.enc_in = nn.Sequential(
                nn.Conv2d(in_ch, h1, 3, padding=1),
                nn.GroupNorm(min(32, h1), h1),
                nn.ReLU(inplace=True),
            )
            self.enc_blocks = nn.Sequential(
                ResBlock(h1), ResBlock(h1), ResBlock(h1),
            )
            # Down
            self.down = nn.Sequential(
                nn.Conv2d(h1, h2, 3, stride=2, padding=1),
                nn.GroupNorm(min(32, h2), h2),
                nn.ReLU(inplace=True),
            )
            # Bottleneck: 2 residual blocks + self-attention
            self.bottleneck = nn.Sequential(
                ResBlock(h2), ResBlock(h2),
                SelfAttention(h2),
            )
            # Up
            self.up = nn.Sequential(
                nn.ConvTranspose2d(h2, h1, 4, stride=2, padding=1),
                nn.GroupNorm(min(32, h1), h1),
            )
            # Decoder: 3 residual blocks + output
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
                h = self.enc_in(x)                       # [B, h1, H, W]
                h = self.enc_blocks(h)                   # 3 residual blocks
                skip = h                                 # save for skip connection
                h = self.down(h)                         # [B, h2, H/2, W/2]
                h = self.bottleneck(h)                   # 2 res blocks + self-attn
                h = F.relu(self.up(h) + skip)            # skip connection
                h = self.dec_blocks(h)                   # 3 residual blocks
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
    """
    Coupling block: splits channels, applies affine transform via s,t,
    reconstructs via r (pseudo-inverse path).
    """
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


def _conv_bn_act(n_in, n_out, **kwargs):
    kwargs["bias"] = False
    khw = 4 if kwargs.get("stride", 1) == 2 else 3
    return nn.Sequential(
        nn.Conv2d(n_in, n_out, khw, padding=1, **kwargs),
        nn.BatchNorm2d(n_out),
        nn.ReLU(inplace=True),
    )


class PatchDiscWithContext(nn.Module):
    """
    Conditional PatchGAN discriminator (from Seraena).
    Takes image + latent context as input. Multi-layer scoring: every block
    produces a 1x1-projected score map, all pooled to a common spatial size
    and concatenated. Uses BatchNorm + ReLU instead of InstanceNorm + LeakyReLU.
    """
    def __init__(self, c_im=3, c_ctx=4):
        super().__init__()
        self.blocks = nn.Sequential(
            _conv_bn_act(c_im + c_ctx, 64),
            _conv_bn_act(64, 128, stride=2),
            _conv_bn_act(128, 256, stride=2),
            _conv_bn_act(256, 512, stride=2),
            _conv_bn_act(512, 512),
            _conv_bn_act(512, 512),
        )
        self.proj = nn.ModuleList(
            nn.Conv2d(block[0].out_channels, 1, 1, bias=False)
            for block in self.blocks
        )

    def forward(self, x, ctx):
        out_hw = tuple(hw // 8 for hw in x.shape[-2:])
        x = torch.cat([x, F.interpolate(ctx, x.shape[-2:], mode="bilinear", align_corners=False)], 1)
        out = []
        for block, proj in zip(self.blocks, self.proj):
            x = block(x)
            out.append(F.adaptive_avg_pool2d(proj(x), out_hw))
        return torch.cat(out, 1)


class SPNNAutoencoder(nn.Module):
    """
    SPNN-based autoencoder for 256x256 images (2-block architecture).

    Encoder path (forward):
        [3, 256, 256]  PixelUnshuffle(4)       -> [48, 64, 64]
        [48, 64, 64]   ConvPINN(48 -> 16)      -> [16, 64, 64]   r: 16->32 (1:2)
        [16, 64, 64]   PixelUnshuffle(2)       -> [64, 32, 32]
        [64, 32, 32]   ConvPINN(64 ->  4)      -> [ 4, 32, 32]   r:  4->60 (1:15)

    Decoder path (pinv): reverses the above using s,t from forward + trained r networks.
    The latent is 4x32x32 = 4096 values, matching the SD-VAE latent shape.
    """
    def __init__(self, mix_type="cayley", hidden=128, r_hidden=256, scale_bound=2.0):
        super().__init__()
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

    def encode(self, x):
        """Encode image -> latent."""
        for b in self.blocks:
            x = b(x)
        return x

    def decode(self, y):
        """Decode latent -> image using pseudo-inverse (r network)."""
        for b in reversed(self.blocks):
            y = b.pinv(y)
        return y


