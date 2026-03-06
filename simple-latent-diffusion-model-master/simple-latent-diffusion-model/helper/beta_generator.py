# https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py#L442
import torch
import math

class BetaGenerator():
    def __init__(self, T) :
        self.T = T

    def fixed_beta_schedule(self, beta) :
        betas = torch.Tensor.repeat(torch.Tensor([beta]) , self.T)
        return betas

    def linear_beta_schedule(self):
        """
        linear schedule, proposed in original ddpm paper
        """
        scale = 1000 / self.T
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return torch.linspace(beta_start, beta_end, self.T)

    def cosine_beta_schedule(self, s = 0.008):
        """
        cosine schedule
        as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
        """
        steps = self.T + 1
        t = torch.linspace(0, self.T, steps, dtype = torch.float32) / self.T
        alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0, 0.999)

    def sigmoid_beta_schedule(self, start = -3, end = 3, tau = 1):
        """
        sigmoid schedule
        proposed in https://arxiv.org/abs/2212.11972 - Figure 8
        better for images > 64x64, when used during training
        """
        steps = self.T + 1
        t = torch.linspace(0, self.T, steps, dtype = torch.float32) / self.T
        v_start = torch.tensor(start / tau).sigmoid()
        v_end = torch.tensor(end / tau).sigmoid()
        alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0, 0.999)