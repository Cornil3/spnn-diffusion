"""
Optimised SPNN autoencoder for 512x512 -> 4x64x64 (SD1.5 VAE latent shape).

Reuses the orthogonal mixes / ResBlock / PixelUnshuffle from spnn_model.py so
the invertible algebra is untouched. What changes is (a) how attention is
computed, (b) how deep each sub-network U-Net is, and (c) the channel-squeeze
ladder.

IDEMPOTENCY is structural and preserved exactly:
    encode: y  = x0*s(x1) + t(x1)
    pinv  : x1 = r(y); x0 = (y - t(x1)) * s(x1)^-1
    => encode(pinv(y)) == y for ANY s,t,r
    => P = decode.encode satisfies P(P(x)) == P(x)
s stays bounded-positive via exp(tanh(.)*bound) and its inverse is the exact
reciprocal (neg=True), so no change here can break invertibility.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from spnn_model import (BaseOrthogonal1x1Conv, Cayley1x1Conv,  # noqa: F401
                        Householder1x1Conv, ResBlock, PixelUnshuffleBlock)



def _groups(ch, maxg=32):
    """Largest divisor of ch that is <= maxg, so GroupNorm never fails."""
    for g in range(min(maxg, ch), 0, -1):
        if ch % g == 0:
            return g
    return 1

class SDPAAttention(nn.Module):
    """Spatial self-attention, same math as spnn_model.SelfAttention but via
    F.scaled_dot_product_attention: O(N) memory instead of materialising the
    [B, HW, HW] score matrix."""

    def __init__(self, channels, n_heads=8):
        super().__init__()
        self.norm = nn.GroupNorm(_groups(channels), channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.out = nn.Conv2d(channels, channels, 1)
        self.n_heads = max(1, min(n_heads, channels // 32))
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        q, k, v = self.qkv(h).chunk(3, dim=1)
        nh = self.n_heads
        # [B, nh, HW, C/nh]
        # .contiguous(): older torch (2.0/2.1) SDPA kernels reject a non-contiguous last
        # dim ('query: last dimension must be contiguous'). Same values, new layout.
        shape = lambda t: t.reshape(B, nh, C // nh, H * W).transpose(-2, -1).contiguous()
        o = F.scaled_dot_product_attention(shape(q), shape(k), shape(v))
        o = o.transpose(-2, -1).reshape(B, C, H, W)
        return x + self.out(o)


class ConvMLPV2(nn.Module):
    """Sub-network for s / t / r. U-Net with `n_down` downsamples; attention
    only at the deepest (cheapest) level."""

    def __init__(self, in_ch, out_ch, scale_bound, hidden_ch=128,
                 feat_size=None, n_down=None, attn=True, max_ch=None):
        super().__init__()
        self.in_ch, self.out_ch, self.scale_bound = in_ch, out_ch, scale_bound

        if in_ch == 0:
            self.mode = "const"
            self.net = nn.Parameter(torch.zeros(1, out_ch, 1, 1))
            return
        if feat_size is not None and feat_size < 4:
            self.mode = "point"
            h = min(max(hidden_ch, in_ch), 512)
            self.net = nn.Sequential(nn.Conv2d(in_ch, h, 1), nn.ReLU(),
                                     nn.Conv2d(h, h, 1), nn.ReLU(),
                                     nn.Conv2d(h, out_ch, 1))
            nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)
            return

        self.mode = "unet"
        # deeper U-Net for larger features -> attention lands at low res and the
        # receptive field grows, which matters at 512 where a fixed RF covers
        # 1/16 the fractional area it did at 128.
        if n_down is None:
            n_down = 2 if (feat_size or 0) >= 64 else 1
        self.n_down = n_down

        h1 = hidden_ch
        cap = max_ch if max_ch is not None else h1 * 2
        chs = [h1] + [min(h1 * (2 ** (i + 1)), cap) for i in range(n_down)]
        self.enc_in = nn.Sequential(
            nn.Conv2d(in_ch, h1, 3, padding=1),
            nn.GroupNorm(_groups(h1), h1), nn.ReLU(inplace=True))
        self.enc_blocks = nn.ModuleList()
        self.downs = nn.ModuleList()
        for i in range(n_down):
            self.enc_blocks.append(nn.Sequential(ResBlock(chs[i]), ResBlock(chs[i])))
            self.downs.append(nn.Sequential(
                nn.Conv2d(chs[i], chs[i + 1], 3, stride=2, padding=1),
                nn.GroupNorm(_groups(chs[i + 1]), chs[i + 1]), nn.ReLU(inplace=True)))
        bott = [ResBlock(chs[-1]), ResBlock(chs[-1])]
        if attn:
            bott.append(SDPAAttention(chs[-1]))
        self.bottleneck = nn.Sequential(*bott)
        self.ups = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        for i in reversed(range(n_down)):
            self.ups.append(nn.Sequential(
                nn.ConvTranspose2d(chs[i + 1], chs[i], 4, stride=2, padding=1),
                nn.GroupNorm(_groups(chs[i]), chs[i])))
            self.dec_blocks.append(nn.Sequential(ResBlock(chs[i]), ResBlock(chs[i])))
        self.out = nn.Conv2d(h1, out_ch, 3, padding=1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, x, neg=False):
        if self.mode == "const":
            B, _, H, W = x.shape
            x = self.net.expand(B, self.out_ch, H, W)
        elif self.mode == "point":
            x = self.net(x)
        else:
            h = self.enc_in(x)
            skips = []
            for blk, dn in zip(self.enc_blocks, self.downs):
                h = blk(h); skips.append(h); h = dn(h)
            h = self.bottleneck(h)
            for up, blk, skip in zip(self.ups, self.dec_blocks, reversed(skips)):
                h = F.relu(up(h) + skip)
                h = blk(h)
            x = self.out(h)

        if self.scale_bound is not None:
            x = torch.tanh(x) * self.scale_bound
            if neg:
                x = -x
            x = x.exp()
        return x


class ConvPINNBlockV2(nn.Module):
    """Identical coupling algebra to spnn_model.ConvPINNBlock; only the
    sub-network class changes. Do not alter forward/pinv - idempotency depends
    on them being exact inverses."""

    def __init__(self, in_ch, out_ch, hidden=128, r_hidden=None, scale_bound=1.0,
                 mix_type="householder", feat_size=None, n_down=None, attn=True,
                 max_ch=None):
        super().__init__()
        assert in_ch > out_ch, f"need in_ch > out_ch (got {in_ch}, {out_ch})"
        self.in_ch, self.out_ch = in_ch, out_ch
        if r_hidden is None:
            r_hidden = hidden * 2
        side_ch = in_ch - out_ch
        self.t = ConvMLPV2(side_ch, out_ch, None, hidden, feat_size, n_down, attn, max_ch)
        self.s = ConvMLPV2(side_ch, out_ch, scale_bound, hidden, feat_size, n_down, attn, max_ch)
        self.r = ConvMLPV2(out_ch, side_ch, None, r_hidden, feat_size, n_down, attn, max_ch)
        self.mix = (Householder1x1Conv(in_ch) if mix_type == "householder"
                    else Cayley1x1Conv(in_ch))

    def forward(self, x):
        x = self.mix.forward(x)
        x0, x1 = x[:, :self.out_ch], x[:, self.out_ch:]
        return x0 * self.s(x1) + self.t(x1)

    def pinv(self, y):
        x1 = self.r(y)
        x0 = (y - self.t(x1)) * self.s(x1, neg=True)
        return self.mix.inverse(torch.cat([x0, x1], dim=1))


class SPNNAutoencoder512Opt(nn.Module):
    """
    [  3, 512, 512]  PixelUnshuffle(4)      -> [ 48, 128, 128]
    [ 48, 128, 128]  ConvPINN(48 -> 24)  2:1 -> [ 24, 128, 128]
    [ 24, 128, 128]  PixelUnshuffle(2)      -> [ 96,  64,  64]
    [ 96,  64,  64]  ConvPINN(96 -> 24)  4:1 -> [ 24,  64,  64]
    [ 24,  64,  64]  ConvPINN(24 ->  4)  6:1 -> [  4,  64,  64]  latent

    f=8, latent 4x64x64 -> drop-in for the SD1.5 VAE latent.
    Squeeze ladder is 2:1 / 4:1 / 6:1 instead of 3:1 / 16:1.
    """

    def __init__(self, mix_type="householder", hidden=96, r_hidden=128,
                 scale_bound=1.0, attn=True, max_ch=192):
        super().__init__()
        mk = lambda i, o, fs, h, rh, nd: ConvPINNBlockV2(
            i, o, hidden=h, r_hidden=rh, scale_bound=scale_bound,
            mix_type=mix_type, feat_size=fs, n_down=nd, attn=attn, max_ch=max_ch)
        self.blocks = nn.ModuleList([
            PixelUnshuffleBlock(4),
            mk(48, 24, 128, hidden, r_hidden, 2),
            PixelUnshuffleBlock(2),
            mk(96, 24, 64, hidden, r_hidden, 2),
            mk(24,  4, 64, hidden, r_hidden, 2),
        ])

    def encode(self, x):
        for b in self.blocks:
            x = b(x)
        return x

    def decode(self, y):
        for b in reversed(self.blocks):
            y = b.pinv(y)
        return y

    def forward(self, x):
        return self.encode(x)
