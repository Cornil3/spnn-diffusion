import torch
import torch.nn as nn

class TextEncoder(nn.Module):
    def __init__(self, number_of_features: int, number_of_heads: int, number_of_transformer_layers: int,
                 context_length, embed_dim):
        super().__init__()
        self.vocab_size = 32000  # AutoTokenizer: "koclip/koclip-base-pt"
        self.token_embedding = nn.Embedding(self.vocab_size, number_of_features)
        self.positional_embedding = nn.Parameter(torch.zeros(context_length, number_of_features))
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=number_of_features, nhead=number_of_heads, batch_first=True),
            num_layers=number_of_transformer_layers
        )
        self.text_projection = nn.Linear(number_of_features, embed_dim)

        # initialize
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.xavier_uniform_(self.positional_embedding)
        nn.init.kaiming_normal_(self.text_projection.weight, nonlinearity='relu')

    def forward(self, x):
        eot_token_idx = (x == 2).nonzero(as_tuple=True)[1]  # Assume EOT token ID is 2
        x = self.token_embedding(x)
        x = x + self.positional_embedding[:x.size(1), :]
        x = self.transformer(x)
        x = x[torch.arange(x.shape[0]), eot_token_idx]
        x = self.text_projection(x)
        return x