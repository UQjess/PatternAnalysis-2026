import torch
from modules import DoubleConv
import torch
from modules import DoubleConv, Down


def test_double_conv():
    block = DoubleConv(1, 64)

    x = torch.randn(4, 1, 256, 144)
    output = block(x)

    assert output.shape == (4, 64, 256, 144)

def test_down():
    block = Down(64, 128)

    x = torch.randn(1, 64, 256, 144)
    output = block(x)

    assert output.shape == (1, 128, 128, 72)