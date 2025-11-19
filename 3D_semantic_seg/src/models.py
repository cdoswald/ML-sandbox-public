"""PyTorch models for Waymo Open Dataset 3D Semantic Segmentation challenge."""

import torch
from torch import nn


class BaselineCNN(nn.Module):
    """Baseline Convolutional Neural Network (CNN) model
    
    Characteristics:
        - Image is padded to maintain spatial dimensions
        - Downsampling using average pooling
        - Normalization using layer norm
        - ReLU activation function
    """

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
        if conv_kernel_size % 2 == 0:
            raise ValueError(f"Convolution kernel size must be an odd integer; got {conv_kernel_size}")
        padding = conv_kernel_size // 2

        # Check that number of layers does not result in spatial dims being downsampled to < 1
        reduc_factor = (avgpool_kernel_size ** len(hidden_layers))
        final_height = self.height / reduc_factor
        final_width = self.width / reduc_factor
        if final_height < 1 or final_width < 1:
            raise ValueError("Too many hidden layers; spatial dims are non-existent by end of model")

        # Create encoder
        encoder_layers = nn.ModuleList()
        curr_in_channels = self.in_channels
        curr_height = self.height
        curr_width = self.width
        for hidden_channels in hidden_layers:
            conv_layer = nn.Conv2d(
                curr_in_channels,
                hidden_channels,
                conv_kernel_size,
                padding=padding,
            )
            layer_norm = nn.LayerNorm((hidden_channels, curr_height, curr_width)) # this might not be optimal for lidar range images
            nonlin_layer = nn.ReLU()
            avgpool_layer = nn.AvgPool2d(avgpool_kernel_size)
            encoder_layers.extend([conv_layer, layer_norm, nonlin_layer, avgpool_layer])
            curr_in_channels = hidden_channels
            curr_height //= avgpool_kernel_size
            curr_width //= avgpool_kernel_size

        self.encoder = nn.Sequential(*encoder_layers)
        print(f"Encoder: {self.encoder}")

        # Create decoder
        flattened_dim = curr_in_channels * curr_height * curr_width
        flatten_layer = nn.Flatten()
        fc1 = nn.Linear(flattened_dim, 256)
        fc2 = nn.Linear(256, out_channels)
        
        self.decoder = nn.Sequential(flatten_layer, fc1, fc2)
        print(f"Decoder: {self.decoder}")


    def forward(self, x):
        encoded_x = self.encoder(x)
        return self.decoder(encoded_x)
