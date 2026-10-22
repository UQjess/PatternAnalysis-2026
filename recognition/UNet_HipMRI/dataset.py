
import os
import glob
import numpy as np
import nibabel as nib

import torch
from torch.utils.data import Dataset, DataLoader


# DATA PREPARATION

# Local computer:
data_dir = r"C:\Users\User\Documents\HipMRI_Data\keras_slices_data"

batch_size = 4

# Directories for training, validation and test sets
train_dir = os.path.join(data_dir, "keras_slices_train")
train_seg_dir = os.path.join(data_dir, "keras_slices_seg_train")

validation_dir = os.path.join(data_dir, "keras_slices_validate")
validation_seg_dir = os.path.join(data_dir, "keras_slices_seg_validate")

test_dir = os.path.join(data_dir, "keras_slices_test")
test_seg_dir = os.path.join(data_dir, "keras_slices_seg_test")

# DATA PREPROCESSING

# Loads MRI images and their segmentation masks
class MRIDataset(Dataset):

    # Find MRI images and their corresponding segmentation masks
    def __init__(self, image_folder, mask_folder):

        self.image_paths = sorted(
            glob.glob(os.path.join(image_folder, "case_*.nii.gz"))
        )

        self.mask_paths = []

        for image_path in self.image_paths:

            mask_name = os.path.basename(image_path).replace(
                "case_", "seg_", 1
            )

            mask_path = os.path.join(mask_folder, mask_name)

            if not os.path.isfile(mask_path):
                raise FileNotFoundError(f"Missing mask: {mask_path}")

            self.mask_paths.append(mask_path)

        if len(self.image_paths) == 0:
            raise ValueError(f"No MRI images found in {image_folder}")

    def __len__(self):
        return len(self.image_paths)

    # Tells PyTorch how to retrieve an MRI image and its mask
    def __getitem__(self, index):

        image = nib.load(
            self.image_paths[index]
        ).get_fdata(dtype=np.float32)

        mask = nib.load(
            self.mask_paths[index]
        ).get_fdata()

        # Normalise MRI intensity values
        image = (image - image.mean()) / (image.std() + 1e-6)

        # Pad images and masks to 256 x 144
        padding = 144 - image.shape[1]

        image = np.pad(
            image,
            ((0, 0), (0, padding)),
            mode="constant"
        )

        mask = np.pad(
            mask,
            ((0, 0), (0, padding)),
            mode="constant"
        )

        # Convert to PyTorch tensors
        image = torch.from_numpy(image.copy()).unsqueeze(0).float()
        mask = torch.from_numpy(mask.copy()).long()

        return image, mask

# Create datasets
train_dataset = MRIDataset(train_dir, train_seg_dir)

validation_dataset = MRIDataset(
    validation_dir,
    validation_seg_dir
)

test_dataset = MRIDataset(test_dir, test_seg_dir)


# Create DataLoaders
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=batch_size,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False
)

# TEST DATA LOADING
if __name__ == "__main__":

    print("Training images:", len(train_dataset))
    print("Validation images:", len(validation_dataset))
    print("Test images:", len(test_dataset))

    images, masks = next(iter(train_loader))

    print("Image shape:", images.shape)
    print("Mask shape:", masks.shape)
    print("Mask labels:", torch.unique(masks))