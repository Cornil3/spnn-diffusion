import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import numpy as np

from clip.encoders.image_encoder import ImageEncoder
from clip.encoders.text_encoder import TextEncoder
from helper.tokenizer import Tokenizer

class CLIP(nn.Module):
    def __init__(self, config_path):
        super().__init__()
        with open(config_path, "r") as file:
           config = yaml.safe_load(file)
           
        self.image_encoder = ImageEncoder(**config["image_encoder"])
        self.text_encoder = TextEncoder(**config["text_encoder"])
        self.tokenizer = Tokenizer()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
        # initialize
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, (nn.LayerNorm, nn.BatchNorm2d)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
                
    def loss(self, image, text):
        image_features, text_features = self(image, text, tokenize=False)

        # Normalize features
        image_features = F.normalize(image_features, dim=1)
        text_features = F.normalize(text_features, dim=1)

        # Cosine similarity as logits with learned temperature
        logits = torch.matmul(image_features, text_features.t()) * self.logit_scale.exp()
        labels = torch.arange(logits.shape[0], dtype=torch.long, device=logits.device)

        # Cross-entropy loss
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.t(), labels)

        return (loss_i2t + loss_t2i) / 2

    def text_encode(self, text, tokenize=True):
        if tokenize:
            tokens = self.tokenizer.tokenize(text)
        else:
            tokens = text
        text_features = self.text_encoder(tokens)
        if text_features.dim() < 2:
            text_features = text_features.unsqueeze(0)
        return text_features
    
    def forward(self, image, text, tokenize=True):
        image_features = self.image_encoder(image)
        text_features = self.text_encoder(text, tokenize)
        
        if image_features.dim() < 2:
            image_features = image_features.unsqueeze(0)
        if text_features.dim() < 2:
            text_features = text_features.unsqueeze(0)
            
        return image_features, text_features