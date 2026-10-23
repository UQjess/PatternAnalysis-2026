
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

        # Decoder upsampling pathway
        self.up4 = UpscaleBlock(128, 64)
        self.local4 = LocalizationBlock(128, 64)

        self.up3 = UpscaleBlock(64, 32)
        self.local3 = LocalizationBlock(64, 32)

        self.up2 = UpscaleBlock(32, 16)
        self.local2 = LocalizationBlock(32, 16)

        self.up1 = UpscaleBlock(16, 8)
        self.local1 = LocalizationBlock(16, 8)

        # Final segmentation output
        self.out_conv = nn.Conv2d(
            8, num_classes,
            kernel_size=1
        )

    def forward(self, x):
        # Extract initial features
        x1, skip1 = self.context1(x)

        # Downsample through the encoder
        x2 = self.context2(x1)
        x3 = self.context3(x2)
        x4 = self.context4(x3)
        x5 = self.context5(x4)

        # Decoder with encoder skip connections
        d4 = self.up4(x5)
        d4 = torch.cat((d4, x4), dim=1)
        d4 = self.local4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat((d3, x3), dim=1)
        d3 = self.local3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat((d2, x2), dim=1)
        d2 = self.local2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat((d1, skip1), dim=1)
        d1 = self.local1(d1)

        return self.out_conv(d1)

