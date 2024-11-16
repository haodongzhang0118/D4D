import torch
import torch.nn as nn
from einops import rearrange

def train_one_batch(inputx, model, device, optimizer):
    # x size: B, T, C, H, W
    inputx = rearrange(inputx, 'b t c h w -> t b c h w')
    indices = torch.stack([torch.randperm(inputx.shape[1]) for _ in range(inputx.shape[0])])  # Shape: [256, 16]
    x = torch.stack([inputx[i, indices[i]] for i in range(inputx.shape[0])])

    T, B, _, _, _ = x.shape
    t = torch.arange(T).to(device)
    for i in range(B):
        images = x[:, i, :, :, :]
        logits = model(images, t)

        optimizer.zero_grad()

        labels = torch.arange(len(images)).to(device)
        loss_i = nn.functional.cross_entropy(logits, labels)
        loss_t = nn.functional.cross_entropy(logits.t(), labels)
        loss = (loss_i + loss_t) / 2

        loss.backward()
        optimizer.step()


