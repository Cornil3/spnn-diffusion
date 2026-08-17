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


def _gn_groups(channels: int, max_groups: int = 32) -> int:
    """Pick number of GroupNorm groups that divides `channels`, capped at max_groups."""
    for g in range(min(max_groups, channels), 0, -1):
        if channels % g == 0:
            return g
    return 1


class _UNet3Skip(nn.Module):
    """4-level U-Net with concat skip connections
    (feat → feat/2 → feat/4 → feat/8 bottleneck → feat/4 → feat/2 → feat).
    Ported from models_deep.py (commit 225ef1f). Used inside ConvMLP as the
    inner s/t/r network when use_deep_convmlp is enabled. Final conv is
    zero-init so the block warm-starts as identity.
    """
    def __init__(self, in_ch, out_ch, h1, h2, h3, h4):
        super().__init__()
        g1, g2, g3, g4 = _gn_groups(h1), _gn_groups(h2), _gn_groups(h3), _gn_groups(h4)

        def _enc(ci, co, g, stride):
            return nn.Sequential(
                nn.Conv2d(ci, co, 3, stride=stride, padding=1), nn.GroupNorm(g, co), nn.ReLU(),
                nn.Conv2d(co, co, 3, padding=1), nn.GroupNorm(g, co), nn.ReLU())

        def _up(ci, co, g):
            return nn.Sequential(
                nn.ConvTranspose2d(ci, co, 4, 2, 1), nn.GroupNorm(g, co), nn.ReLU())

        def _dec(c, g):  # input channels are 2*c after skip-concat
            return nn.Sequential(
                nn.Conv2d(2 * c, c, 3, padding=1), nn.GroupNorm(g, c), nn.ReLU())

        self.enc1, self.enc2 = _enc(in_ch, h1, g1, 1), _enc(h1, h2, g2, 2)
        self.enc3, self.enc4 = _enc(h2, h3, g3, 2), _enc(h3, h4, g4, 2)
        self.up3, self.up2, self.up1 = _up(h4, h3, g3), _up(h3, h2, g2), _up(h2, h1, g1)
        self.dec3, self.dec2, self.dec1 = _dec(h3, g3), _dec(h2, g2), _dec(h1, g1)
        self.final = nn.Conv2d(h1, out_ch, 3, padding=1)
        nn.init.zeros_(self.final.weight)
        nn.init.zeros_(self.final.bias)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        b = self.enc4(e3)
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.final(d1)


class ConvMLP(nn.Module):
    """
    Generic convolutional MLP used for s, t, r sub-networks.
    Builds a small U-Net-like arch for spatial features, or pointwise for 1x1.
    When use_deep_convmlp=True and feat_size >= 8, uses the deep _UNet3Skip
    variant with concat skip connections (bottleneck at feat/8).
    """
    def __init__(self, in_ch, out_ch, scale_bound, hidden_ch=128, feat_size=None,
                 use_deep_convmlp=False):
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
        elif use_deep_convmlp and feat_size is not None and feat_size >= 8 and feat_size % 8 == 0:
            # Deep U-Net inner block with concat skip connections
            # (feat → feat/2 → feat/4 → feat/8 bottleneck → feat/4 → feat/2 → feat).
            # For feat=64 the bottleneck sits at 8x8; for feat=32 it lands at 4x4.
            self._use_unet_deep = True
            h1 = min(max(hidden_ch, in_ch), 1024)
            h2 = min(h1 * 2, 2048)
            h3 = min(h2 * 2, 4096)
            h4 = min(h3 * 2, 8192)
            self.net = _UNet3Skip(in_ch, out_ch, h1, h2, h3, h4)
        elif feat_size is not None and feat_size >= 4:
            # Shallow U-Net (default): single downsample, bottleneck at feat/2, self-attention.
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
            #x = x
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
                 mix_type="cayley", feat_size=None, use_deep_convmlp=False):
        super().__init__()
        assert in_ch > out_ch, (
            f"ConvPINNBlock requires in_ch > out_ch (got {in_ch}, {out_ch})"
        )
        self.in_ch = in_ch
        self.out_ch = out_ch
        if r_hidden is None:
            r_hidden = hidden * 2

        side_ch = in_ch - out_ch
        _mlp = lambda ci, co, sb, hid: ConvMLP(
            ci, co, sb, hid, feat_size=feat_size, use_deep_convmlp=use_deep_convmlp)
        self.t = _mlp(side_ch, out_ch, None, hidden)
        self.s = _mlp(side_ch, out_ch, scale_bound, hidden)
        self.r = _mlp(out_ch, side_ch, None, r_hidden)

        if mix_type == "householder":
            self.mix = Householder1x1Conv(in_ch)
        else:
            self.mix = Cayley1x1Conv(in_ch)

    def forward(self, x):
        x = self.mix.forward(x)
        x0 = x[:, :self.out_ch]
        x1 = x[:, self.out_ch:]
        y = x0 * self.s(x1) + self.t(x1)
        # Store for diagnostics (overwritten each forward call)
        self._diag_x1_true = x1.detach()
        return y

    def pinv(self, y):
        x1 = self.r(y)
        # Store for diagnostics
        self._diag_x1_pred = x1.detach()
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
    def __init__(self, mix_type="cayley", hidden=128, r_hidden=256, scale_bound=2.0,
                 num_blocks=2, use_deep_convmlp=False):
        super().__init__()
        assert num_blocks in (1, 2), f"num_blocks must be 1 or 2, got {num_blocks}"
        self.num_blocks = num_blocks
        _kw = dict(hidden=hidden, r_hidden=r_hidden, scale_bound=scale_bound,
                   mix_type=mix_type, use_deep_convmlp=use_deep_convmlp)
        if num_blocks == 2:
            # Two-stage compression with an intermediate 16x64x64 representation.
            self.blocks = nn.ModuleList([
                # Stage 1: 3x256x256 -> 48x64x64
                PixelUnshuffleBlock(4),
                # Stage 2: 48x64x64 -> 16x64x64
                ConvPINNBlock(48, 16, feat_size=64, **_kw),
                # Stage 3: 16x64x64 -> 64x32x32
                PixelUnshuffleBlock(2),
                # Stage 4: 64x32x32 -> 4x32x32  (latent)
                ConvPINNBlock(64, 4, feat_size=32, **_kw),
            ])
        else:
            # Single-block variant: all downsampling up front, one coupling to the latent.
            # r reconstructs a 188-ch sidechannel from the 4-ch latent (harder inverse).
            self.blocks = nn.ModuleList([
                # Stage 1: 3x256x256 -> 192x32x32
                PixelUnshuffleBlock(8),
                # Stage 2: 192x32x32 -> 4x32x32  (latent)
                ConvPINNBlock(192, 4, feat_size=32, **_kw),
            ])

    def forward(self, x):
        """Forward pass = encode. Required for DDP wrapping."""
        return self.encode(x)

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

    @torch.no_grad()
    def diagnose_r_networks(self, images):
        """
        Run encode then decode on a batch, then print diagnostics for each
        ConvPINNBlock's r network: true x1 range vs predicted x1 range,
        and per-block reconstruction error.

        Usage:
            spnn.eval()
            spnn.diagnose_r_networks(batch_of_images)
        """
        z = self.encode(images)
        x_hat = self.decode(z)

        print(f"\n{'='*70}")
        print(f"SPNN r-network diagnostics  (batch={images.shape[0]})")
        print(f"{'='*70}")
        print(f"Overall roundtrip MSE: {torch.nn.functional.mse_loss(x_hat, images).item():.6f}")
        print()

        for i, block in enumerate(self.blocks):
            if not isinstance(block, ConvPINNBlock):
                continue

            x1_true = block._diag_x1_true
            x1_pred = block._diag_x1_pred

            mse = torch.nn.functional.mse_loss(x1_pred, x1_true).item()
            mae = torch.mean(torch.abs(x1_pred - x1_true)).item()

            # True x1 stats
            t_min = x1_true.min().item()
            t_max = x1_true.max().item()
            t_mean = x1_true.mean().item()
            t_std = x1_true.std().item()
            t_abs_gt1 = (x1_true.abs() > 1.0).float().mean().item() * 100

            # Predicted x1 stats (r output, after tanh)
            p_min = x1_pred.min().item()
            p_max = x1_pred.max().item()
            p_mean = x1_pred.mean().item()
            p_std = x1_pred.std().item()

            print(f"Block {i}: ConvPINNBlock({block.in_ch}→{block.out_ch}), "
                  f"r reconstructs {block.out_ch}→{block.in_ch - block.out_ch} channels")
            print(f"  x1 true  — min={t_min:+.4f}  max={t_max:+.4f}  "
                  f"mean={t_mean:+.4f}  std={t_std:.4f}  |x1|>1: {t_abs_gt1:.1f}%")
            print(f"  x1 pred  — min={p_min:+.4f}  max={p_max:+.4f}  "
                  f"mean={p_mean:+.4f}  std={p_std:.4f}")
            print(f"  r error  — MSE={mse:.6f}  MAE={mae:.4f}")
            print()


