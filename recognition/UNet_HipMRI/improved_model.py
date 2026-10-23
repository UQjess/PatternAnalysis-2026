
import torch
import torch.nn as nn

from modules import (
    FirstContext,
    DownContext,
    UpscaleBlock,
    LocalizationBlock,
)


class ImprovedUNet(nn.Module):
    """Improved 2D U-Net for MRI segmentation."""

    def __init__(self, in_channels=1, num_classes=6):
        super().__init__()

        # Encoder context pathway
        self.context1 = FirstContext(in_channels, 8)
        self.context2 = DownContext(8, 16)
        self.context3 = DownContext(16, 32)
        self.context4 = DownContext(32, 64)
        self.context5 = DownContext(64, 128)

    def forward(self, x):
        # Extract initial features
        x1, skip1 = self.context1(x)

        # Downsample through the encoder
        x2 = self.context2(x1)
        x3 = self.context3(x2)
        x4 = self.context4(x3)
        x5 = self.context5(x4)

        return x5


