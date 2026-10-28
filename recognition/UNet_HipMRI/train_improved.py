import os
import time
import copy

import torch
import torch.nn as nn

from dataset import (
    train_dataset,
    validation_dataset,
    train_loader,
    validation_loader
)

from improved_model import ImprovedUNet

# DICE SCORE

def calculate_dice(predictions, masks, num_classes=6):
    """Calculate intersection and total pixel counts for foreground Dice."""

    intersections = torch.zeros(num_classes)
    totals = torch.zeros(num_classes)

    # Exclude background (class 0) from Dice evaluation
    for class_id in range(1, num_classes):

        predicted_class = predictions == class_id
        actual_class = masks == class_id

        intersections[class_id] = (
            predicted_class & actual_class
        ).sum().item()

        totals[class_id] = (
            predicted_class.sum().item()
            + actual_class.sum().item()
        )

    return intersections, totals

# DEVICE CONFIGURATION

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# HYPERPARAMETERS

num_epochs = 20
learning_rate = 0.001
num_classes = 6


# CREATE IMPROVED MODEL

model = ImprovedUNet(
    in_channels=1,
    num_classes=num_classes
).to(device)


# LOSS AND OPTIMIZER

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=learning_rate
)


# TRAINING

print("> Training")

start = time.time()

# Track the checkpoint with the highest mean foreground validation Dice
best_validation_dice = -1.0
best_model_state = None
best_epoch = 0

checkpoint_path = "checkpoints/unet_improved_best.pth"

for epoch in range(num_epochs):

    model.train()
    total_training_loss = 0

    for images, masks in train_loader:

        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, masks)

        loss.backward()

        optimizer.step()

        total_training_loss += (
            loss.item() * images.size(0)
        )

    average_training_loss = (
        total_training_loss / len(train_dataset)
    )


        # VALIDATION

    model.eval()

    total_validation_loss = 0
    dice_intersections = torch.zeros(num_classes)
    dice_totals = torch.zeros(num_classes)

    with torch.no_grad():

        for images, masks in validation_loader:

            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)

            loss = criterion(outputs, masks)

            total_validation_loss += (
                loss.item() * images.size(0)
            )

            predictions = outputs.argmax(dim=1)

            intersections, totals = calculate_dice(
                predictions,
                masks,
                num_classes
            )

            dice_intersections += intersections
            dice_totals += totals

    average_validation_loss = (
        total_validation_loss / len(validation_dataset)
    )


    # CALCULATE VALIDATION DICE

    class_dice_scores = []

    for class_id in range(1, num_classes):

        if dice_totals[class_id] > 0:

            dice = (
                2 * dice_intersections[class_id]
                / dice_totals[class_id]
            )

            class_dice_scores.append(dice.item())

    average_validation_dice = (
        sum(class_dice_scores) / len(class_dice_scores)
        if class_dice_scores else 0.0
    )


    # SAVE BEST MODEL

    if average_validation_dice > best_validation_dice:

        best_validation_dice = average_validation_dice
        best_epoch = epoch + 1

        best_model_state = copy.deepcopy(
            model.state_dict()
        )

        os.makedirs("checkpoints", exist_ok=True)

        torch.save(
            best_model_state,
            checkpoint_path
        )

    print(
        "Epoch [{}/{}], Training Loss: {:.5f}, "
        "Validation Loss: {:.5f}, Validation Dice: {:.4f}"
        .format(
            epoch + 1,
            num_epochs,
            average_training_loss,
            average_validation_loss,
            average_validation_dice
        )
    )

# TRAINING TIME

if torch.cuda.is_available():
    torch.cuda.synchronize()

elapsed = time.time() - start

print("Training time:", elapsed / 60, "minutes")

# RESTORE BEST TRAINED MODEL

model.load_state_dict(best_model_state)

print("Best epoch:", best_epoch)

print(
    "Best Validation Dice: {:.4f}".format(
        best_validation_dice
    )
)