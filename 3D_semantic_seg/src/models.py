"""PyTorch models for Waymo Open Dataset 3D Semantic Segmentation challenge."""

import torch
from torch import nn


class PlaceholderModel(nn.Module):
    """Doc string"""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return torch.ones_like(x) * 14
