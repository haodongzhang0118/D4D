import torch
import torch.nn as nn
import numpy as np
from vit import vitEncoder
from transformers import CLIPModel, CLIPProcessor

class NoiseEstimationClip(nn.Module):
    def __init__(self, 
                 d_model=768, 
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
        self.temp = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, x, t):
        x = self.vit(x)
        t = self.timestepEncoder(t)
        
        x = x / x.norm(dim=1, keepdim=True)
        t = t / t.norm(dim=1, keepdim=True)
        
        logits = (x @ (t.T)) / self.temp
        
        return logits
    
class NoiseEstimationCLIP_pretrained(nn.Module):
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.clip = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
    def forward(self, images, timesteps):
        self.clip = self.clip.to(self.device)
        texts = [f"This is an image with noise level {t}" for t in timesteps]
        
        # Process inputs
        inputs = self.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True
        )
        
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                 for k, v in inputs.items()}

        outputs = self.clip(**inputs)
        return outputs.logits_per_image
    