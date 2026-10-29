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


def calculate_prostate_metrics(prediction, ground_truth):
    """
    Calculates prostate Dice, IoU and HD95 for a single slice.
    """
    pred = np.asarray(prediction) == PROSTATE_CLASS
    gt = np.asarray(ground_truth) == PROSTATE_CLASS

    intersection = np.logical_and(pred, gt).sum()
    pred_count = pred.sum()
    gt_count = gt.sum()

    # Exclude slices where neither prediction nor ground truth contains prostate
    if pred_count == 0 and gt_count == 0:
        return None

    denominator = pred_count + gt_count
    union = np.logical_or(pred, gt).sum()

    # Dice and IoU measure overlap between predicted and actual prostate regions
    dice = 2 * intersection / denominator if denominator else 1.0
    iou = intersection / union if union else 1.0

    # HD95 cannot be finite if only one mask is empty
    if pred_count == 0 or gt_count == 0:
        hd95 = float("inf")
    else:
        # Extract prostate boundary pixels using binary erosion
        pred_boundary = pred & ~binary_erosion(pred)
        gt_boundary = gt & ~binary_erosion(gt)

        # Calculate distances from each boundary pixel to the opposite boundary
        dist_to_gt = distance_transform_edt(~gt_boundary)
        dist_to_pred = distance_transform_edt(~pred_boundary)

        # Combine distances in both directions for symmetric HD95
        distances = np.concatenate([
            dist_to_gt[pred_boundary],
            dist_to_pred[gt_boundary]
        ])

        # Use the 95th percentile to reduce sensitivity to extreme outliers
        hd95 = float(np.percentile(distances, 95))

    return {
        "dice": float(dice),
        "iou": float(iou),
        "hd95_pixels": hd95,
        "ground_truth_pixels": int(gt_count),
        "predicted_pixels": int(pred_count)
    }

def calculate_metrics(predictions, masks, num_classes=6):
    """
    Calculate pixel counts required for per-class Dice and IoU.
    """

    intersections = torch.zeros(num_classes, dtype=torch.float64)
    predicted_totals = torch.zeros(num_classes, dtype=torch.float64)
    actual_totals = torch.zeros(num_classes, dtype=torch.float64)

    for class_id in range(num_classes):

        predicted_class = predictions == class_id
        actual_class = masks == class_id

        intersections[class_id] = (
            predicted_class & actual_class
        ).sum().item()

        predicted_totals[class_id] = predicted_class.sum().item()
        actual_totals[class_id] = actual_class.sum().item()

    return intersections, predicted_totals, actual_totals


def calculate_scores(intersections, predicted_totals, actual_totals):
    """Calculate per-class Dice and IoU scores."""

    results = []

    for class_id in range(num_classes):

        intersection = intersections[class_id].item()
        predicted = predicted_totals[class_id].item()
        actual = actual_totals[class_id].item()

        # Calculate Dice and IoU denominators
        dice_denominator = predicted + actual
        union = predicted + actual - intersection

        # Avoid division by zero for absent classes
        dice = (
            2 * intersection / dice_denominator
            if dice_denominator > 0 else float("nan")
        )

        iou = (
            intersection / union
            if union > 0 else float("nan")
        )

        results.append({
            "class_id": class_id,
            "dice": dice,
            "iou": iou
        })

    return results


def calculate_mean_foreground(results):
    """Calculate mean Dice and IoU across foreground classes."""

    # Exclude background
    foreground = results[1:]

    # Ignore undefined scores from classes absent in both masks
    valid_dice = [
        result["dice"]
        for result in foreground
        if not torch.isnan(torch.tensor(result["dice"]))
    ]

    valid_iou = [
        result["iou"]
        for result in foreground
        if not torch.isnan(torch.tensor(result["iou"]))
    ]

    mean_dice = (
        sum(valid_dice) / len(valid_dice)
        if valid_dice else float("nan")
    )

    mean_iou = (
        sum(valid_iou) / len(valid_iou)
        if valid_iou else float("nan")
    )

    return mean_dice, mean_iou