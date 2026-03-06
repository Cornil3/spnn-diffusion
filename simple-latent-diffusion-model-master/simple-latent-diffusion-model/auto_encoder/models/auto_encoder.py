import torch.nn as nn
import torch.nn.functional as F
from auto_encoder.models.decoder import Decoder
from auto_encoder.models.encoder import Encoder
import yaml

class AutoEncoder(nn.Module):
    def __init__(self, config_path : str):
        super().__init__()
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        self.add_module('encoder', Encoder(**config["encoder"]))
        self.add_module('decoder', Decoder(**config["decoder"]))
        
    def encode(self, x):
        h = self.encoder(x)
        return h
        
    def decode(self, z):
        z = self.decoder(z)
        return z
    
    def reconstruct(self, x):
        return self.decode(self.encode(x))
    
    def loss(self, x):
        x_hat = self(x)
        return F.mse_loss(x, x_hat)
        
    def forward(self, x):
        return self.reconstruct(x)