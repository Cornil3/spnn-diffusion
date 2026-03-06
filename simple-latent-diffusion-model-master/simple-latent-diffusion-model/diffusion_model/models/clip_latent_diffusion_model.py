import torch
import torch.nn as nn

from auto_encoder.models.variational_auto_encoder import VariationalAutoEncoder
from diffusion_model.models.latent_diffusion_model import LatentDiffusionModel
from clip.models.clip import CLIP

class CLIPLatentDiffusionModel(LatentDiffusionModel) :
    def __init__(self, network : nn.Module, sampler : nn.Module, 
                 auto_encoder : VariationalAutoEncoder, clip : CLIP, image_shape):
        super().__init__(network, sampler, auto_encoder, image_shape)
        self.clip = clip
        self.clip.eval()
        for param in self.clip.parameters():
            param.requires_grad = False
        
    def loss(self, x0, text):
        text = self.clip.text_encode(text, tokenize=False)
        x0 = self.auto_encoder.encode(x0).sample()
        eps = torch.randn_like(x0)
        t = torch.randint(0, self.T, (x0.size(0),), device = x0.device)
        x_t = self.sampler.q_sample(x0, t, eps)
        eps_hat = self.network(x=x_t, t=t, y=text)
        return self.weighted_loss(t, eps, eps_hat)
            
    @torch.no_grad()
    def forward(self, text, n_samples : int = 4):
        text = self.clip.text_encode(text)
        text = text.repeat(n_samples, 1)
        x_T = torch.randn(n_samples, *self.latent_shape, device = next(self.buffers(), None).device )
        sample = self.sampler(x_T = x_T, y=text)
        return self.auto_encoder.decode(sample)

    @torch.no_grad()
    def generate_sequence(self, text, n_samples : int = 4):
        text = self.clip.text_encode(text)
        text = text.repeat(n_samples, 1)
        x_T = torch.randn(n_samples, *self.latent_shape, device = next(self.buffers(), None).device )
        sample_sequence = self.sampler.reverse_process(x_T, y = text, only_last=False)
        return sample_sequence