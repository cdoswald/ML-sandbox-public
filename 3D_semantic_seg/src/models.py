"""PyTorch models for Waymo Open Dataset 3D Semantic Segmentation challenge."""

import torch
from torch import nn


class BaselineCNN(nn.Module):
    """Baseline Convolutional Neural Network (CNN) model
    
    Characteristics:
        - Image is padded to maintain spatial dimensions during convolutions
        - Downsampling (height and width) using average pooling
        - Normalization using 2-D instance norm (distribution for each obs, channel)
        - SiLU activation function
    """

    def __init__(
        self,
        example_batch: torch.Tensor,
        out_channels: int,
        hidden_layers: list[int],
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
            raise ValueError("Too many hidden layers; spatial dims are non-existent after encoding")

        # Create encoder
        encoder_layers = nn.ModuleList()
        curr_in_channels = self.in_channels
        curr_height = self.height
        curr_width = self.width
        for hidden_channels in hidden_layers:
            encode_conv_layer = nn.Conv2d(
                curr_in_channels,
                hidden_channels,
                conv_kernel_size,
                padding=padding,
            )
            norm_layer = nn.InstanceNorm2d(hidden_channels)
            nonlin_layer = nn.SiLU()
            avgpool_layer = nn.AvgPool2d(avgpool_kernel_size)
            encoder_layers.extend(
                [encode_conv_layer, norm_layer, nonlin_layer, avgpool_layer]
            )
            curr_in_channels = hidden_channels
            curr_height //= avgpool_kernel_size
            curr_width //= avgpool_kernel_size

        self.encoder = nn.Sequential(*encoder_layers)
        print(f"Encoder: {self.encoder}")

        # Create decoder (upsample to restore spatial dimension)
        decoder_layers = nn.ModuleList()
        for hidden_channels in reversed(hidden_layers[1:]):
            upsample_layer = nn.Upsample(
                scale_factor=avgpool_kernel_size,
                mode="bilinear",
                align_corners=False,
            )
            decode_conv_layer = nn.Conv2d(
                # TODO: left off here at 10:40pm on 11/20; need to determine channels, height, width 
                # for convolutional layers in decoding block
            )
            
        
        self.decoder = nn.Sequential(*decoder_layers)
        print(f"Decoder: {self.decoder}")


    def forward(self, x):
        encoded_x = self.encoder(x)
        return self.decoder(encoded_x)
