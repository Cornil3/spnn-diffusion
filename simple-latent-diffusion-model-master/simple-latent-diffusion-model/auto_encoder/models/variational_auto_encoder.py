import torch
import torch.nn as nn
import torch.nn.functional as F

from auto_encoder.models.encoder import Encoder
from auto_encoder.models.decoder import Decoder
import yaml
from auto_encoder.components.distributions import DiagonalGaussianDistribution

class VariationalAutoEncoder(nn.Module):
    def __init__(self, config_path):
        super().__init__()
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        self.add_module('encoder', Encoder(**config["encoder"]))
        self.add_module('decoder', Decoder(**config["decoder"]))
        self.embed_dim = config['vae']['embed_dim']
        self.kld_weight = float(config['vae']['kld_weight'])
        
        self.quant_conv = torch.nn.Conv2d(self.decoder.z_channels, 2*self.embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(self.embed_dim, self.decoder.z_channels, 1)

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec
    
    def loss(self, x):
        x_hat, posterior = self(x)
        return F.mse_loss(x, x_hat) + self.kld_weight * posterior.kl().mean() 

    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior