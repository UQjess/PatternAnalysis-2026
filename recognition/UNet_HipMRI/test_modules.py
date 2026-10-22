import torch
from modules import DoubleConv
from modules import DoubleConv, Down, Up


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

def test_up():
    block = Up(1024, 512)

    x1 = torch.randn(1, 1024, 16, 9)
    x2 = torch.randn(1, 512, 32, 18)

    output = block(x1, x2)

    assert output.shape == (1, 512, 32, 18)