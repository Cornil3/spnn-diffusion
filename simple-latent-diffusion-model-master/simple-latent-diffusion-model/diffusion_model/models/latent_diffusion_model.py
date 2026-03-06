import torch
import torch.nn as nn

from auto_encoder.models.variational_auto_encoder import VariationalAutoEncoder
from diffusion_model.models.diffusion_model import DiffusionModel

class LatentDiffusionModel(DiffusionModel) :
    def __init__(self, network : nn.Module, sampler : nn.Module, auto_encoder : VariationalAutoEncoder):
        super().__init__(network, sampler, None)
        self.auto_encoder = auto_encoder
        self.auto_encoder.eval()
        for param in self.auto_encoder.parameters():
            param.requires_grad = False
        # The image shape is the latent shape
        self.image_shape = [*self.auto_encoder.decoder.z_shape[1:]]
        self.image_shape[0] = self.auto_encoder.embed_dim
        
    def loss(self, x0, **kwargs):
        x0 = self.auto_encoder.encode(x0).sample()
        eps = torch.randn_like(x0)
        t = torch.randint(0, self.T, (x0.size(0),), device = x0.device)
        x_t = self.sampler.q_sample(x0, t, eps)
        eps_hat = self.network(x = x_t, t = t, **kwargs)
        return self.weighted_loss(t, eps, eps_hat)

    # The forward function outputs the generated latents
    # Therefore, sample() should be used for sampling data, not latents
    @torch.no_grad()
    def sample(self, n_samples: int = 4, gamma = None, **kwargs):
        sample = self(n_samples, gamma=gamma, **kwargs)
        return self.auto_encoder.decode(sample)
    
    @torch.no_grad()
    def generate_sequence(self, n_samples: int = 4, gamma = None, **kwargs):
        sequence = self(n_samples, only_last=False, gamma = gamma, **kwargs)
        sample = self.auto_encoder.decode(sequence[-1])
        return sequence, sample