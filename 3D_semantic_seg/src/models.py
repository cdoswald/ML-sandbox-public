"""PyTorch models for Waymo Open Dataset 3D Semantic Segmentation challenge."""

import torch
from torch import nn


class BaselineCNN(nn.Module):
    """Baseline Convolutional Neural Network (CNN) Model."""

    def __init__(
        self,
        example_batch: torch.Tensor,
        hidden_layers: list[int],
        out_channels: int,
        conv_kernel_size: int = 5,
        avgpool_kernel_size: int = 2,
    ) -> None:
        super().__init__()
        self.batch_size, self.in_channels, self.height, self.width = example_batch.shape

        # Calculate padding needed to preserve spatial dims
        pad_h = (self.height - 1) // 2
        pad_w = (self.width - 1) // 2

        # Add layers to ModuleList
        self.layers = nn.ModuleList()
        next_in_channels = self.in_channels
        for hidden_channels in hidden_layers:
            conv_layer = nn.Conv2d(
                next_in_channels,
                hidden_channels,
                conv_kernel_size,
                padding=(pad_h, pad_w),
            )
            layer_norm = nn.LayerNorm((hidden_channels, self.height, self.width)) # this might not be the optimal normalization for lidar range images?
            nonlin_layer = nn.ReLU()
            avgpool_layer = nn.AvgPool2d(avgpool_kernel_size)
            self.layers.extend([conv_layer, nonlin_layer, avgpool_layer])
            next_in_channels = hidden_channels
        
        # Add fully connected layers to ModuleList
        
        # Test forward pass/check spatial dims are positive
        
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
