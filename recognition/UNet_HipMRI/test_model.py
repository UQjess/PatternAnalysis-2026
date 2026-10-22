import torch
from model import UNet


def test_unet_output_shape():
    model = UNet(in_channels=1, num_classes=6)
    model.eval()

    x = torch.randn(1, 1, 256, 144)

    with torch.no_grad():
        output = model(x)

    assert output.shape == (1, 6, 256, 144)