# HipMRI Prostate Segmentation

COMP3710 Pattern Recognition

**Project 2**: Segmentation: Anatomical Contouring and Radiotherapy Planning

**Selected model**: 2D Improved U-Net 

**Baseline**: Standard 2D U-Net

## Overview

This project investigates automated anatomical segmentation of MRI images using deep learning. Accurate segmentation is important for medical image analysis, as manually identifying anatomical boundaries can be time-consuming and subject to variability. The project compares a standard 2D U-Net with an improved 2D U-Net to investigate whether architectural modifications can improve segmentation accuracy and boundary precision, particularly for prostate segmentation. Both models are evaluated using identical dataset splits to assess predictive performance, computational efficiency and segmentation failure modes.

## Data and preprocessing

The project used the HipMRI dataset (Dowling et al., 2015), consisting of two-dimensional MRI slices of the male pelvis and their corresponding segmentation masks. The dataset was provided in NIfTI format (.nii.gz) through the `keras_slices_data` directory. Each segmentation mask contains integer labels representing six classes, including the background and five anatomical structures relevant to prostate radiotherapy planning.

Examination of the provided segmentation masks revealed stored label values ranging from 0 to 5, suggesting a zero-based indexing scheme. Based on the classes reported by Dai et al. (2022), the corresponding labels were interpreted as background (0), body (1), bone (2), bladder (3), rectum (4) and prostate (5). This interpretation was further supported by visual inspection of segmentation overlays and consultation with an MRI specialist.

### Dataset exploration and verification

Before model training, the dataset was investigated to understand its structure, label distribution and suitability for segmentation. This included examining MRI dimensions, intensity ranges, image–mask pairing, spatial alignment and patient overlap. Visualisations of the anatomical labels were also examined to verify the interpretation of the segmentation classes.

The data loader pairs images and masks using the patient, week and slice identifiers in their filenames. It checks that image and mask dimensions match and compares their NIfTI affine matrices. The dataset audit also checks that patient identifiers do not occur across different splits.

The dataset split is below.

| Split | MRI Slices | Segmentation Masks | Unique Patients |
|---|---:|---:|---:|
| Training | 11,460 | 11,460 | 31 |
| Validation | 660 | 660 | 4 |
| Testing | 540 | 540 | 3 |
| **Total** | **12,660** | **12,660** | **38** |


Although the test set contains 540 MRI slices, these originate from only three patients, which limits how confidently the results can be generalised to a wider patient population.

### Class distribution

Class imbalance was identified as an important consideration because anatomical structures occupy different proportions of the MRI images. Dai et al. (2022) similarly reported substantial class imbalance in the pelvic MRI dataset, with approximately 270 times more background voxels than prostate voxels. This is  relevant because the prostate is the primary structure of interest, and its relatively small size can make accurate segmentation more challenging.

The class imbalance was assed as

| Class | Pixel Count | Percentage |
|---|---:|---:|
| Class 0 | 111,591,713 | 26.41% |
| Class 1 | 253,682,431 | 60.05% |
| Class 2 | 46,951,538 | 11.11% |
| Class 3 | 7,296,009 | 1.73% |
| Class 4 | 1,340,454 | 0.32% |
| Class 5 | 1,599,295 | 0.38% |

The dataset exhibits substantial class imbalance, with Class 1 (body) accounting for 60.05% of all pixels, while the prostate (Class 5) represents only 0.38%. This imbalance may make the prostate harder to learn. Therefore, prostate-specific Dice and IoU scores are reported alongside overall segmentation performance to assess how effectively each model segments prostate.

### Preprocessing and Augmentation

- **Intensity normalisation:** Each MRI slice is standardised using its mean and standard deviation for consistent intensity scaling.
- **Image padding:** MRI slices and their corresponding masks are padded to 256 × 144 pixels to maintain spatial alignment.
- **Optional one-hot encoding:** The dataset loader supports converting segmentation masks into six one-hot channels. Integer class labels are used during training with multiclass cross-entropy loss.
- **Training augmentation:** Random rotations, translations and intensity scaling are applied during training. Spatial transformations use bilinear interpolation for MRI images and nearest-neighbour interpolation for masks. Augmentation is disabled for validation and testing.

Augmentation was introduced after initial training suggested overfitting; this is a mitigation rather than proof that overfitting was eliminated.

## MODELS

### Standard 2D U-Net

The baseline model follows the standard U-Net architecture proposed by Ronneberger et al. (2015). Ito Aramendia (2024) provided further guidance on the model architecture and implementation, while Milesial’s (2021) PyTorch implementation was used as a reference when developing the model components.

![Standard 2D U-Net](2d-unet.png)

The encoder consists of five stages, increasing the number of feature channels from 64 to 1024. Each stage applies two 3×3 convolutions followed by ReLU activation, with 2×2 max pooling used between stages.

The decoder has four stages using transposed convolutions for upsampling, reducing the number of feature channels from 1024 to 64. Features from matching encoder stages are concatenated with decoder features through skip connections, followed by double-convolution blocks. A final 1 × 1 convolution maps 64 features to six class logits.

The implementation makes two main adaptations to the original U-Net architecture by Ronneberger et al. (2015):

- **Padding:** The 3×3 convolutions use one-pixel padding to preserve spatial dimensions, avoiding the cropping required by the original U-Net's unpadded convolutions.

- **Input and output channels:** The network accepts a single-channel MRI slice and produces six output channels, corresponding to the background and five anatomical segmentation classes.

These adaptations allow the standard U-Net architecture to be applied to the HipMRI dataset while retaining its original encoder–decoder structure.


### Improved 2D U-Net

The improved model is adapted from the modified U-Net architecture proposed by Isensee et al. (2018), originally developed for 3D brain tumour segmentation. The architecture has been adapted to use 2D convolutions for segmentation of the HipMRI dataset. The model was inspired by the BraTS2017 architecture code (Isensee, 2018).

![Improved 2D U-Net](improved-2d-unet.png)

The encoder consists of five stages, increasing the number of feature channels from 8 to 128. Unlike the baseline U-Net, which uses max pooling, the improved model uses stride-2 convolutions for downsampling. Residual context blocks incorporate instance normalisation, LeakyReLU activations (negative slope 0.01) and dropout (p = 0.3).

The decoder consists of four upsampling stages, progressively reducing the number of feature channels from 128 to 8. It uses nearest-neighbour upsampling followed by convolution instead of transposed convolutions. Skip connections concatenate corresponding encoder features with the upsampled decoder features, which are then refined using localisation blocks.

The model also includes two intermediate segmentation prediction heads. Their outputs are upsampled and combined with the final segmentation logits through element-wise addition. A final 1×1 convolution produces six output channels corresponding to the six segmentation classes.
Finally, a 1×1 convolution produces six output channels corresponding to the background and five anatomical segmentation classes.

The implementation makes several adaptations from Isensee et al. (2018):

- **2D adaptation:** Uses 2D convolutions to process individual MRI slices rather than 3D convolutions for volumetric segmentation.

- **Feature channels:** Uses 8–128 channels to reduce model size and computational requirements.

- **Input and output channels:** Accepts single-channel MRI slices and produces six segmentation output channels.

## Training procedure
Rather than reproducing the original training configurations of each architecture, a consistent training procedure was adopted to support a fair comparison. Both the baseline 2D U-Net and improved 2D U-Net were trained on the HipMRI dataset, using identical training and validation splits to ensure a consistent comparison. Both models train using the Adam optimiser with an initial learning rate of 0.001, cross-entropy loss and a batch size of 4. 

Training was conducted for 40 epochs on an NVIDIA A100 GPU using Rangpur, with the training and validation losses monitored after each epoch to assess model convergence and potential overfitting. The model checkpoint achieving the highest validation Dice similarity coefficient (DSC) was saved for subsequent evaluation. 

By maintaining consistent dataset splits, loss function and training hyperparameters consistent, the experiment aims to evaluate the effect of the architectural differences on segmentation performance.

### Training results

| Measure | Baseline U-Net | Improved U-Net |
|---|---:|---:|
| Best epoch | 6 | 14 |
| Best validation foreground Dice | 0.9103 | 0.9077 |
| Training time | 108.65 min | 78.03 min |
| Parameters | 31,030,918 | 796,026 |
| Peak training VRAM | 1.074 GB | 0.225 GB |

Both runs were below the project's proposed 8-hour training and 12 GB GPU-memory limits. The Improved U-Net used approximately 39 times fewer parameters.

The Standard U-Net achieved
its highest validation Dice of 0.9103 at epoch 6, while the Improved
U-Net achieved 0.9077 at epoch 14.

**Standard U-Net**

![Standard U-Net Training and Validation Loss](results/baseline_loss.png)

![Standard U-Net Validation Dice](results/baseline_dice.png)

**Improved U-Net**

![Improved U-Net Training and Validation Loss](results/improved_loss.png)

![Improved U-Net Validation Dice](results/improved_dice.png)

Both models showed decreasing training losses throughout the 40 epochs, indicating they continued learning on the training data. However, validation losses fluctuated and generally increased during later epochs, suggesting potential overfitting. These fluctuations were more pronounced in the Standard U-Net, particularly in its validation loss and Dice scores.

The Standard U-Net achieved a slightly higher peak validation Dice (0.9103) at epoch 6 compared with the Improved U-Net (0.9077) at epoch 14. While the Standard U-Net maintained relatively consistent validation Dice scores overall, it experienced several sharp drops. In contrast, the Improved U-Net showed a more gradual improvement during the early epochs, followed by relatively stable validation Dice scores with smaller fluctuations.

For both models, the checkpoint achieving the highest validation Dice was selected for final evaluation rather.

## Evaluation Procedure
Segmentation accuracy was assessed using the DSC and Intersection over Union (IoU) for each of the six segmentation classes. Mean foreground Dice and IoU scores were also calculated, excluding the background class.

Additional analysis focused on the prostate class, measuring per-slice Dice, IoU and 95th percentile Hausdorff distance (HD95) to assess segmentation overlap and boundary accuracy. To investigate regional performance, prostate-containing slices were divided into three regions representing the two ends and central portion of the prostate. Inter-slice consistency was also examined by measuring changes in predicted segmentation area between adjacent slices.

Segmentation failures were further investigated by identifying disconnected prediction components and generating visual comparisons of MRI images, ground-truth masks and model predictions, including false-positive and false-negative overlays.

Inference time, peak GPU memory usage and model parameter counts were examined to compare the computational efficiency of both models.

## Results

**Per-class segmentation performance on the held-out HipMRI test set.**

| Class | Baseline Dice | Improved Dice | Baseline IoU | Improved IoU |
|---|---:|---:|---:|---:|
| 0 | 0.9978 | 0.9981 | 0.9956 | 0.9962 |
| 1 | 0.9856 | 0.9845 | 0.9716 | 0.9695 |
| 2 | 0.9415 | 0.9377 | 0.8895 | 0.8827 |
| 3 | 0.9663 | 0.9620 | 0.9347 | 0.9269 |
| 4 | 0.8431 | 0.8472 | 0.7287 | 0.7349 |
| **5 (Prostate)** | **0.8563** |**0.8428** | **0.7487** | **0.7282** |

**Overall segmentation performance on the held-out HipMRI test set.**

| Measure | Baseline U-Net | Improved U-Net |
|---|---:|---:|
| Mean foreground Dice (classes 1–5) | 0.9186 | 0.9148 |
| Mean foreground IoU | 0.8547 | 0.8485 |
| **Prostate Dice** | **0.8563** | **0.8428**|
| Prostate IoU | 0.7487 | 0.7282 |
| Mean finite prostate HD95 (pixels) | 3.23 | 3.56 |
| Undefined prostate HD95 cases | 18 | 35 |
| Test loss | 0.0437 | 0.0472 |
| Inference time per image | 9.25 ms | 1.74 ms |
| Peak inference VRAM | 0.421 GB | 0.062 GB |

The baseline U-Net achieved slightly higher overall segmentation performance, with a mean foreground Dice of 0.9186 and IoU of 0.8547, compared with 0.9148 and 0.8485 for the Improved U-Net. For the prostate class, the baseline achieved a Dice score of 0.8563 and IoU of 0.7487, compared with 0.8428 and 0.7282 for the Improved U-Net. Both models exceeded the project's target prostate Dice of 0.75.

The baseline achieved higher Dice and IoU scores for most foreground classes, while the Improved U-Net performed slightly better on Class 4. The baseline also achieved a lower mean finite prostate HD95 (3.23 pixels compared with 3.56 pixels), indicating better boundary agreement among cases with defined HD95 values. However, the Improved U-Net demonstrated greater computational efficiency, achieving an average inference time of 1.74 ms per image compared with 9.25 ms for the baseline, while also using less GPU memory.

### Regional prostate analysis
To examine where segmentation is less reliable, the prostate was divided into approximate apex, mid-gland and base regions using thirds of its superior–inferior extent, identified from the ground-truth prostate masks.

Anatomical orientation was verified using the original 3D HipMRI volumes and NiBabel. A Python verification script compared extracted 2D slices against their corresponding 3D volumes, confirming that the slices were coronal, with increasing column indices representing the superior direction. The verification was documented and repeated across 10 patient/week scans, with all 10 producing exact pixel matches.

Therefore, lower column indices were classified as apex, middle indices as mid-gland and higher indices as base. These regions are approximate rather than independently annotated anatomical landmarks, so the findings remain exploratory. The verification and analysis process was conducted with assistance from OpenAI (2026).

**Regional prostate segmentation performance on the held-out HipMRI test set.**

| Region | Baseline Pooled Dice | Improved Pooled Dice |
|---|---:|---:|
| Apex | 0.8420 | 0.8002 |
| Mid-gland | 0.9102 | 0.8863 |
| Base | 0.8401 | 0.8412 |

Both models achieved their highest regional Dice scores in the mid-gland, with the baseline U-Net scoring 0.9102 and the Improved U-Net scoring 0.8863. Performance was lower towards the apex and base regions, suggesting that segmentation is less reliable at the prostate's anatomical ends.

The baseline U-Net outperformed the Improved U-Net at the apex and mid-gland, while the improved U-Net performed marginally better at the base, although the difference was negligible.

These findings suggest that the Improved U-Net did not consistently improve regional prostate segmentation. The lower Dice scores at the apex and base highlight potential difficulties in segmenting these regions, supporting the need for careful review of predictions near anatomical boundaries.


### Failure case analysis


## Engineering interpretation and recommendation
The Improved U-Net reduces model size, training time, measured inference time, and memory use, but it does not improve prostate Dice or boundary accuracy in this experiment. The standard U-Net is therefore preferable when contour quality is the primary objective under the tested conditions. The Improved U-Net may be useful where compute or memory is severely constrained, but its additional failures require caution.

Neither model is suitable for unsupervised clinical use based on these experiments. A practical annotation assistant should retain expert review, particularly for small prostate regions, uncertain boundaries, disconnected predictions, or absent predictions. Future work could investigate explicit uncertainty estimation, post-processing, stronger validation across patients, and 3D approaches that model inter-slice continuity.
