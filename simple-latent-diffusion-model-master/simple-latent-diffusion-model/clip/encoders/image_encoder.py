import torch
import torch.nn as nn

class ImageEncoder(nn.Module):
    def __init__(self, in_channels: int, resolution: int, patch_size: int, 
                 number_of_features: int, number_of_heads:int, number_of_transformer_layers: int,
                 embed_dim: int):
        super().__init__()
        self.resolution = resolution
        self.embed_dim = embed_dim
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=number_of_features, 
                               kernel_size=patch_size, stride=patch_size, bias=False)
        self.number_of_patches = (resolution // patch_size) ** 2
        self.positional_embedding = nn.Parameter(torch.zeros(1, self.number_of_patches + 1, number_of_features))
        self.class_embedding = nn.Parameter(torch.zeros(1, 1, number_of_features))
        
        self.ln_pre = nn.LayerNorm(number_of_features)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=number_of_features, nhead=number_of_heads, batch_first=True),
            num_layers=number_of_transformer_layers
            )
        
        self.ln_post = nn.LayerNorm(number_of_features)
        self.fc = nn.Linear(number_of_features, embed_dim)
        
        # initialize
        nn.init.kaiming_normal_(self.positional_embedding, nonlinearity='relu')
        nn.init.kaiming_normal_(self.class_embedding, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc.weight, nonlinearity='relu')
        
    def forward(self, x: torch.Tensor):
        x = self.conv(x) # [batch_size, number_of_features, grid, grid]
        x = x.flatten(2) # [batch_size, number_of_features, grid ** 2 = number_of_patches]
        x = x.transpose(1, 2) # [batch_size, number_of_patches, number_of_features]
        
        class_embeddings = self.class_embedding.expand(x.shape[0], -1, -1)
        x = torch.cat([class_embeddings, x], dim=1)
        x = x + self.positional_embedding
        x = self.ln_pre(x)
        x = self.transformer(x) # [batch_size, length_of_sequence, number_of_features]
        x = x.permute(1, 0, 2) # [length_of_sequence, batch_size, number_of_features]
        x = self.ln_post(x[0])
        x = self.fc(x) # [batch_size, embed_dim]
        return x