"""
Evaluates the baseline U-Net and Improved U-Net on the HipMRI test dataset.

Calculates per-class Dice and IoU, prostate-specific Dice, IoU and HD95,
test loss, inference time and peak GPU memory usage.

Saves evaluation results to CSV files for comparison between models.
"""

import sys
import os
import torch
import time
import csv
import torch.nn as nn
import numpy as np

from scipy.ndimage import binary_erosion, distance_transform_edt

from dataset import test_loader
from model import UNet
from improved_model import ImprovedUNet


# Class label representing the prostate in the segmentation masks
PROSTATE_CLASS = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

num_classes = 6
criterion = nn.CrossEntropyLoss()


def load_model(model_type, checkpoint_path):
    """Load a trained U-Net model from a checkpoint."""

    if model_type == "baseline":
        model = UNet(
            in_channels=1,
            num_classes=num_classes
        )

    elif model_type == "improved":
        model = ImprovedUNet(
            in_channels=1,
            num_classes=num_classes
        )

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Restore the best model weights saved during training
    state_dict = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(state_dict)
    model = model.to(device)

    # Switch to evaluation mode
    model.eval()

    return model