import torch

def Normalize(in_channels : int, num_groups : int = 32):
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)