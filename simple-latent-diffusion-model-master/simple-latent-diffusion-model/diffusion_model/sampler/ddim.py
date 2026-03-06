import torch

from diffusion_model.sampler.base_sampler import BaseSampler

class DDIM(BaseSampler):
    def __init__(self, config_path):
        super().__init__(config_path)
        self.sampling_T = self.config['sampling_T']
        step = self.T // self.sampling_T
        self.timesteps = torch.arange(0, self.T, step, dtype=torch.int)

        self.ddim_alpha = self.alpha_bar[self.timesteps]
        self.sqrt_one_minus_alpha_bar = (1. - self.ddim_alpha).sqrt()
        self.alpha_bar_prev = torch.cat([self.ddim_alpha[0:1], self.ddim_alpha[:-1]])
        self.sigma = (self.config['eta'] * 
                            torch.sqrt((1-self.alpha_bar_prev) / (1-self.ddim_alpha) *
                            (1 - self.ddim_alpha / self.alpha_bar_prev)))

    def get_x_prev(self, x, tau, eps_hat) :
        alpha_prev = self.alpha_bar_prev[tau]
        sigma = self.sigma[tau]

        x0_hat = (x - self.sqrt_one_minus_alpha_bar[tau] * eps_hat) \
           / (self.ddim_alpha[tau] ** 0.5)
        dir_xt = (1. - alpha_prev - sigma ** 2).sqrt() * eps_hat
        if sigma == 0. : noise = 0.
        else : noise = torch.randn_like(x, device = x.device)
        x = alpha_prev.sqrt() * x0_hat + dir_xt + sigma * noise
        return x