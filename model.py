import torch
import torch.nn as nn
import numpy as np
from vit import vitEncoder
from timestep_embedding import TimestepEmbedding

class NoiseEstimationClip(nn.Module):
    def __init__(self, 
                 d_model=768, 
                 specific_timesteps=256, 
                 in_channels=3, 
                 image_size=256, 
                 patch_size=16, 
                 num_heads=8, 
                 num_layers=12,
                 final_embedding=768):
        super().__init__()
        self.vit = vitEncoder(d_model=d_model, 
                              in_channels=in_channels, 
                              image_size=image_size, 
                              patch_size=patch_size, 
                              num_heads=num_heads,
                              num_layers=num_layers,
                              final_embedding_dim=final_embedding)
        self.timestepEncoder = vitEncoder(d_model=d_model, 
                              in_channels=in_channels, 
                              image_size=image_size, 
                              patch_size=patch_size, 
                              num_heads=num_heads,
                              num_layers=num_layers,
                              final_embedding_dim=final_embedding)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, x, t):
        x = self.vit(x)
        t = self.timestepEncoder(t)
        
        x = x / x.norm(dim=1, keepdim=True)
        t = t / t.norm(dim=1, keepdim=True)
        
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * x @ t.t()
        
        return logits