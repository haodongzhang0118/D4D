import torch
import torch.nn as nn
from einops import rearrange

class Blindspot(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)
        self.mask = torch.ones((in_channels, in_channels, 3, 3))
        self.mask[:, :, 1, 1] = 0
        self.mask = nn.Parameter(self.mask, requires_grad=False)
    
    def forward(self, x):
        self.conv.weight.data *= self.mask
        return self.conv(x)

class PatchEmbeddings(nn.Module):
    def __init__(self, d_model, patch_size, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.conv(x)
        B, C, H, W = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        return x
    
class LearnablePositionalEmbeddings(nn.Module):
    def __init__(self, d_model, patchs):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.zeros(1, patchs + 1, d_model), requires_grad=True)

    def forward(self, x):
        return x + self.pos_embedding
    
class vitEncoder(nn.Module):
    def __init__(self, 
                 d_model=768, 
                 in_channels=3, 
                 image_size=256, 
                 patch_size=16, 
                 num_heads=8, 
                 num_layers=12,
                 final_embedding_dim=768):
        super().__init__()
        self.blindspot = Blindspot(in_channels)
        patch_dim = 3 * (image_size // patch_size) ** 2
        self.patch_embed = PatchEmbeddings(patch_dim, patch_size, in_channels)
        self.pos_embed = LearnablePositionalEmbeddings(d_model, (image_size // patch_size) ** 2)
        self.mlp = nn.Linear(patch_dim, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        encoderLayer = nn.TransformerEncoderLayer(d_model, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoderLayer, num_layers=num_layers)

        self.fc = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, final_embedding_dim)
        )

    def forward(self, x):
        B = x.shape[0]
        x = self.blindspot(x)
        x = self.patch_embed(x)
        x = self.mlp(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_embed(x)
        x = self.transformer(x)
        x = x[:, 0] # B D
        x = self.fc(x) # B F
        return x


    
