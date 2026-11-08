"""Gated recurrent attention kernel for tabular sequence forecasting."""

import torch
import torch.nn as nn


class GatedRecurrentAttentionKernel(nn.Module):
    def __init__(self, features=32, heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(features, heads, batch_first=True)
        self.gate = nn.Linear(features, features)

    def forward(self, x):
        out, _ = self.attn(x, x, x)
        return out * torch.sigmoid(self.gate(out))
