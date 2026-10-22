import torch
from modules import DoubleConv


def test_double_conv():
    block = DoubleConv(1, 64)

    x = torch.randn(4, 1, 256, 144)
    output = block(x)

    assert output.shape == (4, 64, 256, 144)