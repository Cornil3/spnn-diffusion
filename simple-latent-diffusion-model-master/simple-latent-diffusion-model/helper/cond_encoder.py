import torch
import torch.nn as nn
import yaml

class BaseCondEncoder(nn.Module):
    def __init__(
        self, 
        config_path
        ):
        super().__init__()
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)['cond_encoder']
        self.embed_dim = self.config['embed_dim']
        self.cond_dim = self.config['cond_dim']
        if 'cond_drop_prob' in self.config:
            self.cond_drop_prob = self.config['cond_drop_prob']
            self.null_embedding = nn.Parameter(torch.randn(self.embed_dim))
        else:
            self.cond_drop_prob = 0.0

        self.cond_mlp = nn.Sequential(
            nn.Linear(self.embed_dim, self.cond_dim),
            nn.GELU(),
            nn.Linear(self.cond_dim, self.cond_dim)
            )

    def cond_drop(self, y: torch.tensor):
        if self.training and self.cond_drop_prob > 0.0:
            flags = torch.zeros((y.size(0), ), device=y.device).float().uniform_(0, 1) < self.cond_drop_prob
            y[flags] = self.null_embedding.to(y.dtype)
        return y

class CLIPEncoder(BaseCondEncoder):
    def __init__(
        self,
        clip,
        config_path
        ):
        super().__init__(config_path)
        self.clip = clip
        self.clip.eval()
        for param in self.clip.parameters():
            param.requires_grad = False

    def forward(self, y, cond_drop_all:bool = False):
        if isinstance(y, str):
            y = self.clip.text_encode(y, tokenize=True)
        else:
            y = self.clip.text_encode(y, tokenize=False)
        y = self.cond_drop(y) # Only training
        if cond_drop_all:
            y[:] = self.null_embedding
        return self.cond_mlp(y)

class ClassEncoder(BaseCondEncoder):
    def __init__(
        self, 
        config_path
        ):
        super().__init__(config_path)
        self.num_cond = self.config['num_cond']
        self.embed = nn.Embedding(self.num_cond, self.embed_dim)

    def forward(self, y, cond_drop_all:bool = False):
        y = self.embed(y)
        y = self.cond_drop(y) # Only training
        if cond_drop_all:
            y[:] = self.null_embedding
        return self.cond_mlp(y)