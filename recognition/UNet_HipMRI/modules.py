import torch
import torch.nn as nn
import torch.nn.functional as F


# STANDARD U-NET

class DoubleConv(nn.Module):
    """Applies two 3x3 convolutions, each followed by ReLU."""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        # First convolution layer
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1
        )

        # Second convolution layer
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1
        )

    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        return x

class Down(nn.Module):
    """Downsamples using max pooling followed by double convolution."""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upsamples and combines encoder features using a skip connection."""
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_channels,
            in_channels // 2,
            kernel_size=2,
            stride=2
        )

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        # Upsample the decoder feature map
        x1 = self.up(x1)

        # Combine with encoder feature map
        x = torch.cat([x2, x1], dim=1)

        # Apply two convolutions
        return self.conv(x)

class OutConv(nn.Module):
    """Applies a 1x1 convolution to produce segmentation class scores."""
    def __init__(self, in_channels, out_channels):
        super().__init__()

        # Convert feature channels into the number of segmentation classes
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1
        )

    def forward(self, x):
        return self.conv(x)