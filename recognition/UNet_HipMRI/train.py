
import os
import time
import copy

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from dataset import (
    train_dataset,
    validation_dataset,
    train_loader,
    validation_loader
)

from model import UNet


# DEVICE CONFIGURATION

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# HYPERPARAMETERS

num_epochs = 30
learning_rate = 0.001
num_classes = 6


# CREATE MODEL

model = UNet(
    in_channels=1,
    num_classes=num_classes
).to(device)


# LOSS AND OPTIMIZER

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=learning_rate
)
