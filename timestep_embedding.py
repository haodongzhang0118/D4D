import torch
import torch.nn as nn

class TimestepEmbedding(nn.Module):
    def __init__(self, d_model=768, specific_timesteps=256, final_embedding=768):
        super().__init__()
        self.embedding = nn.Embedding(specific_timesteps, d_model)
        self.fc = nn.Sequential(
            nn.GELU(),
            nn.Linear(d_model, final_embedding)
        )

    def forward(self, x):
        return self.fc(self.embedding(x))