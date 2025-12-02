"""PyTorch models for Waymo Open Dataset 3D Semantic Segmentation challenge."""

import math

import torch
from torch import nn


class BaselineCNN(nn.Module):
    """Baseline Convolutional Neural Network (CNN) model
    
    Characteristics:
        - Image is padded to maintain spatial dimensions during convolutions
        - Spatial downsampling using average pooling
        - Spatial upsampling using bilinear interpolation
        - Normalization using 2-D instance norm (distribution for each obs, channel)
        - SiLU activation function
    """

    def __init__(
        self,
        example_batch: torch.Tensor,
        n_classes: int,
        conv_layers: list[int],
        avgpool_layers: list[tuple[int, int]],
        conv_kernel_size: int = 5,
    ) -> None:
        super().__init__()
        self.batch_size, self.in_channels, self.height, self.width = example_batch.shape

        # Calculate padding needed to preserve spatial dims
        if conv_kernel_size % 2 == 0:
            raise ValueError(f"Convolution kernel size must be an odd integer; got {conv_kernel_size}")
        padding = conv_kernel_size // 2

        # Check that number of avgpool layers matches the number of convolutional layers
        if len(conv_layers) != len(avgpool_layers):
            raise ValueError(f"Arguments 'conv_layers' and 'avgpool_layers' must be same length")

        # Check that final height and width are integers
        height_reduc_factor = math.prod([x[0] for x in avgpool_layers])
        width_reduc_factor = math.prod([x[1] for x in avgpool_layers])
        final_height = float(self.height / height_reduc_factor)
        final_width = float(self.width / width_reduc_factor)
        if final_height < 1 or final_width < 1:
            raise ValueError(f"Final height and width must be >= 1; got ({final_height}, {final_width})")
        if not final_height.is_integer() or not final_width.is_integer():
            raise ValueError(f"Final height and width must be integers; got ({final_height}, {final_width})")

        # Create encoder
        encoder_layers = nn.ModuleList()
        curr_in_channels = self.in_channels

        hidden_layers=[128, 256, 512, 1024], #TODO: remove
        avgpool_layers=[(1,5), (1,5), (1,2), (2,1)] #TODO: remove
        
        for conv_channels, avgpool_kernel in zip(conv_layers, avgpool_layers):
            encoder_conv_layer = nn.Conv2d(
                curr_in_channels,
                conv_channels,
                conv_kernel_size,
                padding=padding,
            )
            encoder_norm_layer = nn.InstanceNorm2d(conv_channels)
            encoder_activ_layer = nn.SiLU()
            encoder_downsample_layer = nn.AvgPool2d(avgpool_kernel)
            encoder_layers.extend([
                encoder_conv_layer,
                encoder_norm_layer,
                encoder_activ_layer,
                encoder_downsample_layer
            ])
            curr_in_channels = conv_channels

        self.encoder = nn.Sequential(*encoder_layers)
        print(f"Encoder: {self.encoder}")

        # Create decoder (upsample to restore original spatial dimensions)
        decoder_layers = nn.ModuleList()
        # TODO: left off here on 12/1 at 11:00pm; need to figure out how to align convolution channels
        # and upsampling operations (currently off by 1)
        for conv_channels, avgpool_kernel in zip(reversed(conv_layers), reversed(avgpool_layers)):
            decoder_upsample_layer = nn.Upsample(
                scale_factor=avgpool_kernel,
                mode="bilinear",
                align_corners=False,
            )
            decoder_conv_layer = nn.Conv2d(
                curr_in_channels,
                conv_channels,
                conv_kernel_size,
                padding=padding,
            )
            decoder_norm_layer = nn.InstanceNorm2d(conv_channels)
            decoder_activ_layer = nn.SiLU()
            decoder_layers.extend([
                decoder_upsample_layer,
                decoder_conv_layer,
                decoder_norm_layer,
                decoder_activ_layer,
            ])
            curr_in_channels = conv_channels

        logit_head = nn.Conv2d(
            curr_in_channels,
            n_classes,
            kernel_size=1,
        )
        decoder_layers.append(logit_head)

        self.decoder = nn.Sequential(*decoder_layers)
        print(f"Decoder: {self.decoder}")


    def forward(self, x: torch.tensor):
        encoded_x = self.encoder(x)
        return self.decoder(encoded_x)
