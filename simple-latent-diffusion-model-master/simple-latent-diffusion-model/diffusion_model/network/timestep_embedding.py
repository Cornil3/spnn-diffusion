import torch
import torch.nn as nn
import math

class SinusoidalEmbedding(nn.Module):
    def __init__(self, embed_dim : int, theta : int = 10000):
        """
        Creates sinusoidal embeddings for timesteps.

        Args:
            embed_dim: The dimensionality of the embedding.
            theta: The base for the log-spaced frequencies.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.theta = theta

    def forward(self, x):
        """
        Computes sinusoidal embeddings for the input timesteps.

        Args:
            x: A 1D torch.Tensor of timesteps (shape: [batch_size]).

        Returns:
            A torch.Tensor of sinusoidal embeddings (shape: [batch_size, embed_dim]).
        """
        assert isinstance(x, torch.Tensor) # Input must be a torch.Tensor
        assert x.ndim == 1 # Input must be a 1D tensor
        assert isinstance(self.embed_dim, int) and self.embed_dim > 0 # embed_dim must be a positive integer

        half_dim = self.embed_dim // 2
        # Create a sequence of log-spaced frequencies
        embeddings = math.log(self.theta) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=x.device) * -embeddings)
        # Outer product: timesteps x frequencies
        embeddings = x[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        # Handle odd embedding dimensions
        if self.embed_dim % 2 == 1:
            embeddings = torch.cat([embeddings, torch.zeros_like(embeddings[:, :1])], dim=-1)
        return embeddings