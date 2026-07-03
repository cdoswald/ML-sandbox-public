"""PyTorch models for Waymo Open Dataset 3D Semantic Segmentation challenge."""

import math

import torch
import torch.nn as nn


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
        hidden_channels: list[int],
        avgpool_layers: list[tuple[int, int]],
        conv_kernel_size: int = 5,
        repeat_conv_n_times: int = 3,
        verbose: bool = False,
    ) -> None:

        super().__init__()
        self.batch_size, self.in_channels, self.height, self.width = example_batch.shape

        # Calculate padding needed to preserve spatial dims
        if conv_kernel_size % 2 == 0:
            raise ValueError(
                f"Convolution kernel size must be an odd integer; got {conv_kernel_size}"
            )
        padding = conv_kernel_size // 2

        # Check that correct number of avgpool layers provided
        n_avgpool_layers = len(avgpool_layers)
        n_expected_avgpool_layers = len(hidden_channels) - 1
        if n_avgpool_layers > n_expected_avgpool_layers:
            raise ValueError(
                f"Too many 'avgpool_layers' provided; expected {n_expected_avgpool_layers} "
                + f"but got {n_avgpool_layers}"
            )
        if n_avgpool_layers < n_expected_avgpool_layers:
            raise ValueError(
                f"Too few 'avgpool_layers' provided; expected {n_expected_avgpool_layers} "
                + f"but got {n_avgpool_layers}"
            )

        # Check that final height and width are integers
        height_reduc_factor = math.prod([x[0] for x in avgpool_layers])
        width_reduc_factor = math.prod([x[1] for x in avgpool_layers])
        final_height = float(self.height / height_reduc_factor)
        final_width = float(self.width / width_reduc_factor)
        if final_height < 1 or final_width < 1:
            raise ValueError(
                f"Final height and width must be >= 1; got ({final_height}, {final_width})"
            )
        if not final_height.is_integer() or not final_width.is_integer():
            raise ValueError(
                f"Final height and width must be integers; got ({final_height}, {final_width})"
            )

        # Create encoder layers
        encoder_layers = nn.ModuleList()
        curr_in_channels = self.in_channels
        for i, curr_hidden_channels in enumerate(hidden_channels):
            # Conv block
            for _ in range(repeat_conv_n_times):
                encoder_conv_layer = nn.Conv2d(
                    curr_in_channels,
                    curr_hidden_channels,
                    conv_kernel_size,
                    padding=padding,
                )
                encoder_norm_layer = nn.InstanceNorm2d(curr_hidden_channels)
                encoder_activ_layer = nn.SiLU()
                encoder_layers.extend(
                    [
                        encoder_conv_layer,
                        encoder_norm_layer,
                        encoder_activ_layer,
                    ]
                )
                # Update in-channels
                curr_in_channels = curr_hidden_channels
            # Downsample (for all but last encoder block)
            if i < len(hidden_channels) - 1:
                encoder_layers.append(nn.AvgPool2d(avgpool_layers[i]))

        # Create decoder layers
        decoder_layers = nn.ModuleList()
        rev_hidden_channels = list(reversed(hidden_channels[:-1]))
        upsample_layers = list(reversed(avgpool_layers))
        for j, curr_hidden_channels in enumerate(rev_hidden_channels):
            # Upsample
            decoder_layers.append(
                nn.Upsample(
                    scale_factor=upsample_layers[j],
                    mode="bilinear",
                    align_corners=False,
                )
            )
            # Conv block
            for _ in range(repeat_conv_n_times):
                decoder_conv_layer = nn.Conv2d(
                    curr_in_channels,
                    curr_hidden_channels,
                    conv_kernel_size,
                    padding=padding,
                )
                decoder_norm_layer = nn.InstanceNorm2d(curr_hidden_channels)
                decoder_activ_layer = nn.SiLU()
                decoder_layers.extend(
                    [
                        decoder_conv_layer,
                        decoder_norm_layer,
                        decoder_activ_layer,
                    ]
                )
                # Update in-channels
                curr_in_channels = curr_hidden_channels

        # Add classification head
        logit_head = nn.Conv2d(
            curr_in_channels,
            n_classes,
            kernel_size=1,
        )
        decoder_layers.append(logit_head)

        # Unpack layers to Sequential
        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)

        # Print model details
        if verbose:
            print(f"Encoder: {self.encoder}")
            print(f"Decoder: {self.decoder}")

    def forward(self, x: torch.tensor):
        encoded_x = self.encoder(x)
        return self.decoder(encoded_x)
