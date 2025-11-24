"""PyTorch models for Waymo Open Dataset 3D Semantic Segmentation challenge."""

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
        # and final height and width are integers
        reduc_factor = (avgpool_kernel_size ** len(hidden_layers))
        final_height = float(self.height / reduc_factor)
        final_width = float(self.width / reduc_factor)
        if final_height < 1 or final_width < 1:
            raise ValueError("Too many hidden layers; spatial dims downspampled to < 1 during encoding")
        if not final_height.is_integer() or not final_width.is_integer():
            raise ValueError("Input height and width must be divisible by avgpool_kernel_size ** num_layers")

        # Create encoder
        encoder_layers = nn.ModuleList()
        curr_in_channels = self.in_channels

        for hidden_channels in hidden_layers:
            encoder_conv_layer = nn.Conv2d(
                curr_in_channels,
                hidden_channels,
                conv_kernel_size,
                padding=padding,
            )
            encoder_norm_layer = nn.InstanceNorm2d(hidden_channels)
            encoder_activ_layer = nn.SiLU()
            encoder_downsample_layer = nn.AvgPool2d(avgpool_kernel_size)
            encoder_layers.extend([
                encoder_conv_layer,
                encoder_norm_layer,
                encoder_activ_layer,
                encoder_downsample_layer
            ])
            curr_in_channels = hidden_channels

        self.encoder = nn.Sequential(*encoder_layers)
        print(f"Encoder: {self.encoder}")

        # Create decoder (upsample to restore spatial dimension)
        decoder_layers = nn.ModuleList()
        for hidden_channels in reversed(hidden_layers[:-1]):
            decoder_upsample_layer = nn.Upsample(
                scale_factor=avgpool_kernel_size,
                mode="bilinear",
                align_corners=False,
            )
            decoder_conv_layer = nn.Conv2d(
                curr_in_channels,
                hidden_channels,
                conv_kernel_size,
                padding=padding,
            )
            decoder_norm_layer = nn.InstanceNorm2d(hidden_channels)
            decoder_activ_layer = nn.SiLU()
            decoder_layers.extend([
                decoder_upsample_layer,
                decoder_conv_layer,
                decoder_norm_layer,
                decoder_activ_layer,
            ])
            curr_in_channels = hidden_channels

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
