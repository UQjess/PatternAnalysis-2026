
import os
import glob
import numpy as np
import nibabel as nib

import torch
from torch.utils.data import Dataset, DataLoader


# DATA PREPARATION

# local computer:
data_dir = r"C:\Users\User\Documents\HipMRI_Data\keras_slices_data"

batch_size = 4

# Directories for training, validation and test sets
train_dir = os.path.join(data_dir, "keras_slices_train")
train_seg_dir = os.path.join(data_dir, "keras_slices_seg_train")

validation_dir = os.path.join(data_dir, "keras_slices_validate")
validation_seg_dir = os.path.join(data_dir, "keras_slices_seg_validate")

test_dir = os.path.join(data_dir, "keras_slices_test")
test_seg_dir = os.path.join(data_dir, "keras_slices_seg_test")
