# https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py

from functools import partial
import torch
from torch import nn
from torch.nn import Module, ModuleList
from diffusion_model.network.attention import LinearAttention, Attention
from diffusion_model.network.timestep_embedding import SinusoidalEmbedding
from diffusion_model.network.blocks import ResnetBlock

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def cast_tuple(t, length = 1):
    if isinstance(t, tuple):
        return t
    return ((t,) * length)

def divisible_by(numer, denom):
    return (numer % denom) == 0

# small helper modules

class DownSample(nn.Module):
    def __init__(self, dim: int, dim_out: int):
        """
        Downsamples the spatial dimensions by a factor of 2 using a strided convolution.

        Args:
            dim: Input channel dimension.
        """
        super().__init__()
        self.downsample = nn.Conv2d(dim, dim_out, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape [B, C, H, W].

        Returns:
            Downsampled tensor of shape [B, C, H/2, W/2].
        """
        return self.downsample(x)

class UpSample(nn.Module):
    def __init__(self, dim: int, dim_out: int):
        """
        Upsamples the spatial dimensions by a factor of 2 using a transposed convolution.

        Args:
            dim: Input channel dimension.
        """
        super().__init__()
        self.upsample = nn.ConvTranspose2d(dim, dim_out, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape [B, C, H, W].

        Returns:
            Upsampled tensor of shape [B, C, 2*H, 2*W].
        """
        return self.upsample(x)


# model

class Unet(Module):
    def __init__(
        self,
        dim,
        init_dim = None,
        out_dim = None,
        cond_dim = None,
        dim_mults = (1, 2, 4, 8),
        channels = 3,
        dropout = 0.,
        attn_dim_head = 32,
        attn_heads = 4,
        full_attn = None,    # defaults to full attention only for inner most layer
    ):
        super().__init__()

        # determine dimensions

        self.channels = channels
        input_channels = channels

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding = 3)

        dims = [*map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # time embeddings
        time_dim = dim * 4

        sinu_pos_emb = SinusoidalEmbedding(dim)

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # attention

        if not full_attn:
            full_attn = (*((False,) * (len(dim_mults) - 1)), True)

        num_stages = len(dim_mults)
        full_attn  = cast_tuple(full_attn, num_stages)
        attn_heads = cast_tuple(attn_heads, num_stages)
        attn_dim_head = cast_tuple(attn_dim_head, num_stages)

        assert len(full_attn) == len(dim_mults)

        # prepare blocks

        FullAttention = Attention
        resnet_block = partial(ResnetBlock, 
                               t_emb_dim = time_dim, y_emb_dim = cond_dim, dropout = dropout)

        # layers

        self.downs = ModuleList([])
        self.ups = ModuleList([])
        num_resolutions = len(in_out)

        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(zip(in_out, full_attn, attn_heads, attn_dim_head)):
            is_last = ind >= (num_resolutions - 1)

            attn_klass = FullAttention if layer_full_attn else LinearAttention

            self.downs.append(ModuleList([
                resnet_block(dim_in, dim_in),
                resnet_block(dim_in, dim_in),
                attn_klass(dim_in, dim_head = layer_attn_dim_head, heads = layer_attn_heads),
                DownSample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding = 1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = resnet_block(mid_dim, mid_dim)
        self.mid_attn = FullAttention(mid_dim, heads = attn_heads[-1], dim_head = attn_dim_head[-1])
        self.mid_block2 = resnet_block(mid_dim, mid_dim)

        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(zip(*map(reversed, (in_out, full_attn, attn_heads, attn_dim_head)))):
            is_last = ind == (len(in_out) - 1)

            attn_klass = FullAttention if layer_full_attn else LinearAttention

            self.ups.append(ModuleList([
                resnet_block(dim_out + dim_in, dim_out),
                resnet_block(dim_out + dim_in, dim_out),
                attn_klass(dim_out, dim_head = layer_attn_dim_head, heads = layer_attn_heads),
                UpSample(dim_out, dim_in) if not is_last else  nn.Conv2d(dim_out, dim_in, 3, padding = 1)
            ]))

        default_out_dim = channels
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = resnet_block(init_dim * 2, init_dim)
        self.final_conv = nn.Conv2d(init_dim, self.out_dim, 1)

    @property
    def downsample_factor(self):
        return 2 ** (len(self.downs) - 1)

    def forward(self, x, t, y = None):
        assert all([divisible_by(d, self.downsample_factor) for d in x.shape[-2:]]), f'your input dimensions {x.shape[-2:]} need to be divisible by {self.downsample_factor}, given the unet'

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(t)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t, y)
            h.append(x)

            x = block2(x, t, y)
            x = attn(x) + x
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t, y)
        x = self.mid_attn(x) + x
        x = self.mid_block2(x, t, y)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x, t, y)

            x = torch.cat((x, h.pop()), dim = 1)
            x = block2(x, t, y)
            x = attn(x) + x

            x = upsample(x)

        x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x, t, y)
        return self.final_conv(x)