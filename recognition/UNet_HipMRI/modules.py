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


class FirstContext(nn.Module):
    """First encoder context block of the Improved 2D U-Net."""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        # Initial convolution to extract features
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1,
            bias=False
        )

        # Additional convolutions for residual feature learning
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1,
            bias=False
        )

        self.conv3 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1,
            bias=False
        )

        # Activation, regularisation, and normalisation
        self.activation = nn.LeakyReLU(negative_slope=0.01)
        self.dropout = nn.Dropout(p=0.3)
        self.norm = nn.InstanceNorm2d(out_channels)

    def forward(self, x):
        # Save initial features for residual connection
        out = self.conv1(x)
        residual = out

        # Apply convolutions with activation and dropout
        out = self.activation(out)
        out = self.conv2(out)
        out = self.dropout(out)
        out = self.activation(out)
        out = self.conv3(out)

        # Add residual features
        out = out + residual

        # Save activated features for the decoder skip connection
        skip = self.activation(out)

        # Normalise features for the next encoder level
        out = self.norm(out)
        out = self.activation(out)

        return out, skip


