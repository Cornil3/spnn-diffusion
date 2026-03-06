import torch
import torch.nn as nn

class AdaptiveGroupNorm(nn.Module):
    def __init__(self, num_groups, num_channels, emb_dim, eps=1e-5):
        super().__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        # Use a standard GroupNorm, but without learnable affine parameters
        self.norm = nn.GroupNorm(num_groups, num_channels, eps=eps, affine=False)

        # Linear layers to project the embedding to gamma and beta
        self.gamma_proj = nn.Linear(emb_dim, num_channels)
        self.beta_proj = nn.Linear(emb_dim, num_channels)
        
    def forward(self, x, emb):
        """
        Args:
            x: Input tensor of shape [B, C, H, W].
            emb: Embedding tensor of shape [B, emb_dim].

        Returns:
            Normalized tensor with adaptive scaling and shifting.
        """
        # Normalize as usual with GroupNorm
        normalized = self.norm(x)

        # Get gamma and beta from the embedding
        gamma = self.gamma_proj(emb)
        beta = self.beta_proj(emb)

        # Reshape for broadcasting: [B, C] -> [B, C, 1, 1]
        gamma = gamma.view(-1, self.num_channels, 1, 1)
        beta = beta.view(-1, self.num_channels, 1, 1)

        # Apply adaptive scaling and shifting
        return gamma * normalized + beta

class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, padding):
        super().__init__()
        self.depthwise = nn.Conv2d(dim_in, dim_in, kernel_size, padding=padding, groups=dim_in)
        self.pointwise = nn.Conv2d(dim_in, dim_out, 1)  # 1x1 convolution

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups, emb_dim, dropout=0.0, use_depthwise=False):
        super().__init__()
        self.norm = AdaptiveGroupNorm(groups, dim, emb_dim)
        if use_depthwise:
            self.proj = DepthwiseSeparableConv2d(dim, dim_out, kernel_size=3, padding=1)
        else:
            self.proj = nn.Conv2d(dim, dim_out, 3, padding=1)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, emb):
        x = self.norm(x, emb)  # Pre-normalization
        x = self.proj(x)
        x = self.act(x)
        return self.dropout(x)

class ResnetBlock(nn.Module):
    def __init__(self, dim: int, dim_out: int, t_emb_dim: int, *,
                y_emb_dim: int = None, groups: int = 32, dropout: float = 0.0, residual_scale=1.0):
        super().__init__()
        if y_emb_dim is None:
            y_emb_dim = 0
        emb_dim = t_emb_dim + y_emb_dim

        self.block1 = Block(dim, dim_out, groups, emb_dim, dropout)  # Pass emb_dim
        self.block2 = Block(dim_out, dim_out, groups, emb_dim, dropout) # Pass emb_dim
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()
        self.residual_scale = nn.Parameter(torch.tensor(residual_scale))

    def forward(self, x, t_emb, y_emb=None):
        cond_emb = t_emb
        if y_emb is not None:
            cond_emb = torch.cat([cond_emb, y_emb], dim=-1)

        h = self.block1(x, cond_emb)  # Pass combined embedding to Block
        h = self.block2(h, cond_emb)  # Pass combined embedding to Block

        return self.residual_scale * h + self.res_conv(x)  # Scale the residual