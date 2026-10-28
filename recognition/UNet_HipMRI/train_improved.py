import os
import time

import torch
import torch.nn as nn

from dataset import (
    train_dataset,
    validation_dataset,
    train_loader,
    validation_loader
)

from improved_model import ImprovedUNet


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

    with torch.no_grad():

        for images, masks in validation_loader:

            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)

            loss = criterion(outputs, masks)

            total_validation_loss += (
                loss.item() * images.size(0)
            )

    average_validation_loss = (
        total_validation_loss / len(validation_dataset)
    )

    print(
        "Epoch [{}/{}], Training Loss: {:.5f}, "
        "Validation Loss: {:.5f}"
        .format(
            epoch + 1,
            num_epochs,
            average_training_loss,
            average_validation_loss
        )
    )


# TRAINING TIME

if torch.cuda.is_available():
    torch.cuda.synchronize()

elapsed = time.time() - start

print("Training time:", elapsed / 60, "minutes")