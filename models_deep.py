import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn_groups(channels: int, max_groups: int = 32) -> int:
    """Largest divisor of `channels` that is <= max_groups, for nn.GroupNorm."""
    for g in (max_groups, 16, 8, 4, 2, 1):
        if channels % g == 0:
            return g
    return 1


class _UNet3Skip(nn.Module):
    """3-level U-net with concat skip connections (feat → /2 → /4 → /8 → /4 → /2 → feat).
    Final conv is zero-init so the block warm-starts as identity, matching the
    Sequential branches' init contract.
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
        e1 = self.enc1(x);
        e2 = self.enc2(e1);
        e3 = self.enc3(e2);
        b = self.enc4(e3)
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.final(d1)


class BaseOrthogonal1x1Conv(nn.Module):
    """
    Base class for orthogonal 1x1 convolutions.

    Subclasses must implement:
        _compute_W(device, dtype) -> [C, C] orthogonal matrix
    """

    def __init__(self, channels):
        super().__init__()
        self.channels = channels

    def _compute_W(self, device, dtype):
        raise NotImplementedError

    def forward(self, x):
        B, C, H, W = x.shape
        assert C == self.channels, f"Expected {self.channels} channels, got {C}"

        W = self._compute_W(x.device, x.dtype)  # [C, C]
        weight = W.view(C, C, 1, 1)  # [C_out, C_in, 1, 1]
        return F.conv2d(x, weight)

    def inverse(self, x):
        B, C, H, W = x.shape
        assert C == self.channels, f"Expected {self.channels} channels, got {C}"

        W = self._compute_W(x.device, x.dtype)  # [C, C]
        W_inv = W.t()
        weight = W_inv.view(C, C, 1, 1)
        return F.conv2d(x, weight)


class Householder1x1Conv(BaseOrthogonal1x1Conv):
    """
    Orthogonal 1x1 conv via product of Householder reflections:
        W = H_k ... H_1, H_i = I - 2 v_i v_i^T / ||v_i||^2

    Forward and inverse apply reflections directly to x (rank-1 updates)
    without materializing the full W matrix, giving machine-precision
    accuracy at any channel count.
    """

    def __init__(self, channels, num_reflections=8, eps=1e-8):
        super().__init__(channels)
        self.num_reflections = num_reflections
        self.eps = eps

        if num_reflections > 0:
            V = torch.randn(num_reflections, channels)
            self.V = nn.Parameter(V)
        else:
            self.register_parameter("V", None)

    def _get_normalized_V(self, device, dtype):
        """Return normalized reflection vectors [K, C]."""
        V = self.V.to(device=device, dtype=dtype)
        return V / (V.norm(p=2, dim=1, keepdim=True) + self.eps)

    def _compute_W(self, device, dtype):
        """Materialize full W matrix (for compatibility). Prefer forward/inverse."""
        C = self.channels
        if self.V is None or self.num_reflections == 0:
            return torch.eye(C, device=device, dtype=dtype)
        W = torch.eye(C, device=device, dtype=dtype)
        V_norm = self._get_normalized_V(device, dtype)
        for i in range(self.num_reflections):
            v = V_norm[i]
            H = torch.eye(C, device=device, dtype=dtype) - 2.0 * torch.outer(v, v)
            W = H @ W
        return W

    def forward(self, x):
        """Apply W = H_k ... H_1 to x via sequential rank-1 updates."""
        if self.V is None or self.num_reflections == 0:
            return x
        B, C, H, W = x.shape
        V_norm = self._get_normalized_V(x.device, x.dtype)  # [K, C]
        # Reshape to [B, C, H*W] for batched dot products
        y = x.reshape(B, C, -1)
        # Apply H_1, then H_2, ..., H_k (forward order)
        for i in range(self.num_reflections):
            v = V_norm[i]  # [C]
            dots = torch.einsum('c,bcn->bn', v, y)  # [B, H*W]
            y = y - 2.0 * v[None, :, None] * dots[:, None, :]
        return y.reshape(B, C, H, W)

    def inverse(self, x):
        """Apply W^T = H_1 ... H_k to x (reverse order, since H_i^T = H_i)."""
        if self.V is None or self.num_reflections == 0:
            return x
        B, C, H, W = x.shape
        V_norm = self._get_normalized_V(x.device, x.dtype)  # [K, C]
        y = x.reshape(B, C, -1)
        # Apply H_k, then H_{k-1}, ..., H_1 (reverse order)
        for i in range(self.num_reflections - 1, -1, -1):
            v = V_norm[i]
            dots = torch.einsum('c,bcn->bn', v, y)
            y = y - 2.0 * v[None, :, None] * dots[:, None, :]
        return y.reshape(B, C, H, W)


class BasePatchOrthogonalMix(nn.Module):
    """
    Base class for orthogonal and invertible patch wise mixer.

    Pipeline (shared for all subclasses):
      - Unfold image into non overlapping patches of size p×p
      - Flatten each patch to a vector of size D = c_in * p * p
      - Apply the same orthogonal W ∈ R^{D×D} to every patch
      - Fold back to image

    Subclasses must implement:
      _compute_W(device, dtype) returns [D, D] orthogonal matrix
    """

    def __init__(self, in_ch, patch_size=4):
        super().__init__()
        self.in_ch = in_ch
        self.patch_size = patch_size
        self.D = in_ch * patch_size * patch_size  # patch vector dim

        # will take non overlapping patches of size patch_size×patch_size and flatten them
        self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)

    def _compute_W(self, device, dtype):
        """
        need to return an orthogonal matrix W ∈ R^{D×D}
        """
        raise NotImplementedError

    def forward(self, x):
        B, C, H, W = x.shape
        assert C == self.in_ch, f"Expected {self.in_ch} channels, got {C}"
        assert H % self.patch_size == 0 and W % self.patch_size == 0, \
            "H and W must be divisible by patch_size"

        patches = self.unfold(x)  # [B=batch_size, D=c_in * p * p, L=num of patches per images]
        B_, D, L = patches.shape
        patches = patches.transpose(1, 2)  # [B, L, D] – take row vectors

        W_mat = self._compute_W(x.device, x.dtype)  # [D, D]

        # Forward: apply W^T on row vectors
        patches_mixed = patches @ W_mat.T  # [B, L, D]
        patches_mixed = patches_mixed.transpose(1, 2)  # [B, D, L]

        fold = nn.Fold(
            output_size=(H, W),
            kernel_size=self.patch_size,
            stride=self.patch_size
        )
        y = fold(patches_mixed)  # back to [B, C, H, W]

        return y

    def inverse(self, x):
        B, C, H, W = x.shape
        assert C == self.in_ch, f"Expected {self.in_ch} channels, got {C}"
        assert H % self.patch_size == 0 and W % self.patch_size == 0, \
            "H and W must be divisible by patch_size"

        patches = self.unfold(x)  # [B, D, L]
        B_, D, L = patches.shape
        patches = patches.transpose(1, 2)  # [B, L, D]

        W_mat = self._compute_W(x.device, x.dtype)  # [D, D]

        # Inverse: apply W (since forward used W^T)
        patches_unmixed = patches @ W_mat  # [B, L, D]
        patches_unmixed = patches_unmixed.transpose(1, 2)  # [B, D, L]

        fold = nn.Fold(
            output_size=(H, W),
            kernel_size=self.patch_size,
            stride=self.patch_size
        )
        y = fold(patches_unmixed)  # [B, C, H, W]
        return y

class PatchHouseholderMix(BasePatchOrthogonalMix):
    """
    Orthogonal, invertible patch-wise mixer using Householder reflections.
    """

    def __init__(self, in_ch, patch_size=2, num_reflections=4, eps=1e-8):
        super().__init__(in_ch, patch_size)
        self.num_reflections = num_reflections
        self.eps = eps

        if num_reflections > 0:
            V = torch.randn(num_reflections, self.D)
            self.V = nn.Parameter(V)
        else:
            self.register_parameter("V", None)

    def _compute_W(self, device, dtype):
        D = self.D
        if self.V is None or self.num_reflections == 0:
            return torch.eye(D, device=device, dtype=dtype)
        W = torch.eye(D, device=device, dtype=dtype)
        V = self.V.to(device=device, dtype=dtype)
        for i in range(self.num_reflections):
            v = V[i]
            v = v / (v.norm(p=2) + self.eps)
            H = torch.eye(D, device=device, dtype=dtype) - 2.0 * torch.outer(v, v)
            W = H @ W
        return W  # [D, D]


class ConvMLP(nn.Module):
    def __init__(self, in_ch, out_ch, scale_bound, hidden_ch, img_size: int = 32,
                 feat_size: int = None):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.scale_bound = scale_bound
        self.img_size = img_size

        if in_ch > 0:
            h1 = min(max(hidden_ch, in_ch), 1024)
            h2 = min(h1 * 2, 2048)
            h3 = min(h2 * 2, 4096)
            if feat_size is not None and feat_size >= 8 and feat_size % 8 == 0:
                # Deeper U-net: 4 levels with concat skip connections
                # (feat → feat/2 → feat/4 → feat/8 bottleneck → feat/4 → feat/2 → feat).
                # Skip connections concatenate encoder activations into the
                # matching decoder stage. Bijectivity of the enclosing
                # ConvPINNBlock is preserved: s/t/r are arbitrary deterministic
                # functions used inside y = x0*s(x1) + t(x1) — the inverse
                # re-uses the same s/t outputs. Final conv is zero-init inside
                # _UNet3Skip so the warm-start "head ≈ identity" is intact.
                h4 = min(h3 * 2, 8192)
                self.net = _UNet3Skip(in_ch, out_ch, h1, h2, h3, h4)
            elif feat_size is not None and feat_size > 1:
                assert feat_size % 2 == 0, (
                    f"feat_size must be even when using the stride-2 path (got feat_size={feat_size}). "
                    f"ConvTranspose2d(kernel=4, stride=2, pad=1) restores H→H only for even H."
                )
                # Shallow U-net: 2 levels (for feat_size=2 only)
                self.net = nn.Sequential(
                    nn.Conv2d(in_ch, h1, 3, padding=1), nn.ReLU(),
                    nn.Conv2d(h1, h1, 3, padding=1), nn.ReLU(),
                    nn.Conv2d(h1, h2, 3, stride=2, padding=1), nn.ReLU(),
                    nn.Conv2d(h2, h2, 3, padding=1), nn.ReLU(),
                    nn.ConvTranspose2d(h2, h1, 4, stride=2, padding=1), nn.ReLU(),
                    nn.Conv2d(h1, out_ch, 3, padding=1),
                )
            elif feat_size == 1:
                # 1x1 spatial: use pointwise convs like the tailored 1x1 cases
                self.net = nn.Sequential(
                    nn.Conv2d(in_ch, h1, 1), nn.ReLU(),
                    nn.Conv2d(h1, h2, 1), nn.ReLU(),
                    nn.Conv2d(h2, h1, 1), nn.ReLU(),
                    nn.Conv2d(h1, out_ch, 1),
                )
            else:
                # feat_size unknown:
                self.net = nn.Sequential(
                    nn.Conv2d(in_ch, h1, 3, padding=1), nn.ReLU(),
                    nn.Conv2d(h1, h2, 3, padding=1), nn.ReLU(),
                    nn.Conv2d(h2, h2, 3, padding=1), nn.ReLU(),
                    nn.Conv2d(h2, h1, 3, padding=1), nn.ReLU(),
                    nn.Conv2d(h1, out_ch, 3, padding=1),
                )
            if isinstance(self.net, nn.Sequential):
                nn.init.zeros_(self.net[-1].weight)
                nn.init.zeros_(self.net[-1].bias)
            # _UNet3Skip handles its own zero-init internally.

        # If in_ch == 0, treat it as a learned constant bias per output channel
        else:
            self.net = nn.Parameter(torch.zeros(1, out_ch, 1, 1))

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
        return x


class PixelUnshuffleBlock(nn.Module):
    def __init__(self, r: int):
        super().__init__()
        self.r = r

    def forward(self, x, return_latent=False):
        y = F.pixel_unshuffle(x, self.r)
        return y, None

    def pinv(self, y, x1_override=None):
        return F.pixel_shuffle(y, self.r)


class PINN(nn.Module):
    def __init__(self, block_cls, layer_channels, img_size: int = 64, num_classes=40, mix_type: str = "cayley",
                 **block_kwargs):
        super().__init__()

        if layer_channels is not None:
            # DIY network: build blocks from user-provided block_cls and layer_channels.
            # Checked first so that layer_channels always overrides built-in architectures.
            #
            # Each entry in layer_channels must be one of:
            #   (BlockClass, kwargs_dict)  – per-block class + constructor kwargs
            #   kwargs_dict                – constructor kwargs using the shared block_cls
            #
            # Example:
            #   layer_channels = [
            #       (PixelUnshuffleBlock, {"r": 4}),                              # [3,H,W] -> [48,H/4,W/4]
            #       (ConvPINNBlock, {"in_ch": 48, "out_ch": 12, "hidden": 128, "scale_bound": 2.0}),
            #       (ConvPINNBlock, {"in_ch": 12, "out_ch": num_classes, "hidden": 128, "scale_bound": 2.0}),
            #   ]
            if not layer_channels:
                raise ValueError("layer_channels must be a non-empty list for custom architectures")

            blocks = []
            for spec in layer_channels:
                if isinstance(spec, (list, tuple)) and len(spec) == 2 and isinstance(spec[1], dict):
                    cls, kwargs = spec
                elif isinstance(spec, dict):
                    if block_cls is None:
                        raise ValueError(
                            "block_cls must be provided when layer_channels entries are plain dicts"
                        )
                    cls, kwargs = block_cls, spec
                else:
                    raise ValueError(
                        f"Each layer_channels entry must be a (BlockClass, kwargs_dict) tuple "
                        f"or a kwargs dict (with block_cls set), got {type(spec)}"
                    )
                # block_kwargs are shared defaults; per-block kwargs take priority
                merged = {**block_kwargs, **kwargs}
                blocks.append(cls(**merged))
            self.blocks = nn.ModuleList(blocks)

        elif img_size == 64:
            self.blocks = nn.ModuleList([
                PixelUnshuffleBlock(2),  # [3,64,64] -> [12,32,32]
                ConvPINNBlock(12, 6, hidden=128, scale_bound=2.0, mix_type=mix_type),
                ConvPINNBlock(6, 3, hidden=128, scale_bound=2.0, mix_type=mix_type),

                PixelUnshuffleBlock(4),  # [3,32,32] -> [48,8,8]
                ConvPINNBlock(48, 32, hidden=128, scale_bound=2.0, mix_type=mix_type),

                PixelUnshuffleBlock(8),  # [32,8,8] -> [2048,1,1]
                ConvPINNBlock(2048, 1024, hidden=128, scale_bound=2.0, mix_type=mix_type),

                ConvPINNBlock(1024, num_classes, hidden=128, scale_bound=2.0, mix_type=mix_type),
            ])
        elif img_size == 256:
            self.blocks = nn.ModuleList([
                PixelUnshuffleBlock(4),  # [3,256,256] -> [48,64,64]
                ConvPINNBlock(48, 12, hidden=128, scale_bound=2.0, img_size=256, mix_type=mix_type),

                PixelUnshuffleBlock(4),  # [12,64,64] -> [192,16,16]
                ConvPINNBlock(192, 48, hidden=128, scale_bound=2.0, img_size=256, mix_type=mix_type),

                PixelUnshuffleBlock(4),  # [48,16,16] -> [768,4,4]
                ConvPINNBlock(768, 192, hidden=128, scale_bound=2.0, img_size=256, mix_type=mix_type),

                PixelUnshuffleBlock(4),  # [192,4,4] -> [3072,1,1]
                ConvPINNBlock(3072, 1024, hidden=128, scale_bound=2.0, img_size=256, mix_type=mix_type),

                ConvPINNBlock(1024, num_classes, hidden=128, scale_bound=2.0, img_size=256, mix_type=mix_type),
            ])
        elif img_size == 32:
            self.blocks = nn.ModuleList([
                PixelUnshuffleBlock(4),  # [3,32,32] -> [48,8,8]
                ConvPINNBlock(48, 32, hidden=128, scale_bound=2.0, mix_type=mix_type),

                PixelUnshuffleBlock(8),  # [32,8,8] -> [2048,1,1]
                ConvPINNBlock(2048, 1024, hidden=128, scale_bound=2.0, mix_type=mix_type),

                ConvPINNBlock(1024, num_classes, hidden=128, scale_bound=2.0, mix_type=mix_type),
            ])
        else:
            raise ValueError(
                f"img_size must be 32, 64 or 256 for built-in architectures (got {img_size}). "
                "To use a custom size, pass your own block definitions via layer_channels."
            )

    def forward(self, x, return_latents=False):
        latents = []

        for b in self.blocks:
            # UNIFIED INTERFACE: Every block guarantees a return of (y, z)
            x, z = b(x, return_latent=return_latents)
            latents.append(z)
        return (x, latents) if return_latents else x

    def pinv(self, y, latents=None):
        if latents is not None:
            z_stack = list(reversed(latents))
        else:
            z_stack = [None] * len(self.blocks)

        for b in reversed(self.blocks):
            z = z_stack.pop(0)
            y = b.pinv(y, x1_override=z)
        return y


class ConvPINNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, hidden=64, scale_bound=2., img_size: int = 32, mix_type: str = "householder",
                 feat_size: int = None):
        super().__init__()
        assert in_ch > out_ch, (
            f"ConvPINNBlock requires in_ch > out_ch (got in_ch={in_ch}, out_ch={out_ch}). "
            f"The coupling layer splits channels as x0=[:out_ch], x1=[out_ch:], so in_ch must be strictly larger."
        )

        layers = {
            "t": (in_ch - out_ch, out_ch, None, hidden),
            "s": (in_ch - out_ch, out_ch, scale_bound, hidden),
            "r": (out_ch, in_ch - out_ch, None, hidden)
        }

        for name, (in_s, out_s, sb, hid) in layers.items():
            setattr(self, name, ConvMLP(in_s, out_s, sb, hid, img_size=img_size, feat_size=feat_size))

        if mix_type == "householder":
            self.mix = Householder1x1Conv(in_ch)
        else:
            self.mix = Householder1x1Conv(in_ch)
        self.in_ch = in_ch
        self.out_ch = out_ch

    def forward(self, x, return_latent=False):
        x = self.mix.forward(x)
        x0 = x[:, :self.out_ch, :, :]
        x1 = x[:, self.out_ch:, :, :]
        y = x0 * self.s(x1) + self.t(x1)
        z = x1 if return_latent else None
        return y, z

    def pinv(self, y, x1_override=None):
        x1 = self.r(y) if x1_override is None else x1_override
        x0 = (y - self.t(x1)) * self.s(x1, neg=True)
        x = torch.cat([x0, x1], dim=1)
        return self.mix.inverse(x)


class SPNN(nn.Module):
    def __init__(
            self,
            img_ch: int = 3,
            num_classes: int = 40,
            hidden: int = 128,
            scale_bound: float = 2.0,
            img_size: int = 64,
            mix_type: str = "cayley",
            block_cls=None,
            layer_channels=None,
            output_spatial_size=None,
            **block_kwargs,
    ):
        super().__init__()
        if layer_channels is None:
            assert img_size in (32, 64, 256), (
                f"img_size must be 32, 64 or 256 for built-in architectures (got {img_size}). "
                "To use a custom size, pass your own block definitions via layer_channels."
            )
        if output_spatial_size is None:
            assert num_classes < 1024, "num of classes (output size) must be less then 1024"
        self.img_ch = img_ch
        self.num_classes = num_classes
        self.hidden = hidden
        self.scale_bound = scale_bound
        self.img_size = img_size
        self.output_spatial_size = output_spatial_size

        self.pinn = PINN(block_cls=block_cls, layer_channels=layer_channels, img_size=img_size, mix_type=mix_type,
                         **block_kwargs)

    def forward(self, x_img, return_latents=False):
        B, C, H, W = x_img.shape
        assert C == self.img_ch

        out = self.pinn(x_img, return_latents=return_latents)
        if return_latents:
            y_map, latents = out
        else:
            y_map = out

        if self.output_spatial_size is None:
            logits = y_map.view(B, self.num_classes)
        else:
            logits = y_map

        return (logits, latents) if return_latents else logits

    def pinv(self, logits, latents=None):
        if self.output_spatial_size is None:
            B, C = logits.shape
            assert C == self.num_classes
            y_map_hat = logits.view(B, self.num_classes, 1, 1)
        else:
            y_map_hat = logits

        return self.pinn.pinv(y_map_hat, latents=latents)


class PixelShuffleLayer(nn.Module):
    def __init__(self, upscale_factor: int):
        super().__init__()
        self.r = upscale_factor

    def forward(self, x):
        return F.pixel_shuffle(x, self.r)
