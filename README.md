# Distortion Classification for IQA Dataset Analysis

This project trains an image distortion classifier on the **TID2013** image quality assessment dataset and applies the classifier to other IQA datasets such as **KonIQ10K** and **SPAQ**. The goal is to identify dominant distortion types in real-world image quality datasets and use the predicted distortion categories to support IQA model analysis, data cleaning, and failure-case interpretation.

## Motivation

Image Quality Assessment (IQA) models often produce inaccurate predictions when the test images contain complex or underrepresented distortions. Instead of only evaluating IQA scores, this project analyzes the **type of visual distortion** present in low-quality images.

By classifying images into distortion categories, we can answer questions such as:

- What distortion types appear most frequently in low-MOS images?
- Are certain distortion categories harder for IQA models to evaluate?
- Can distortion classification help explain IQA model failure cases?
- Can predicted distortion labels support data cleaning or dataset analysis?

## Project Overview

The pipeline contains three main stages:

1. **Train a distortion classifier on TID2013**
   - Uses the 24 official TID2013 distortion categories.
   - Trains a ConvNeXt-Tiny based classifier.
   - Saves the best model checkpoint based on validation accuracy.

2. **Filter low-quality images from IQA datasets**
   - Selects images whose MOS score is below a threshold.
   - Supports datasets such as SPAQ and KonIQ10K.

3. **Predict distortion categories**
   - Applies the trained classifier to filtered images.
   - Outputs predicted distortion labels, confidence scores, and top-3 predictions.
   - Generates category statistics and visualization plots.

## Method

### Distortion Classification Model

The classifier uses a pretrained **ConvNeXt-Tiny** backbone from `timm`, followed by a custom classification head:

- ConvNeXt-Tiny feature extractor
- Linear layer with 1024 hidden units
- GELU activation
- Dropout regularization
- Final linear layer for 24 distortion classes

The model is trained using:

- Cross-entropy loss
- AdamW optimizer
- Cosine annealing learning-rate scheduler
- ImageNet normalization
- Random resized crop, horizontal flip, and color jitter for training augmentation

### Distortion Classes

The model follows the 24 distortion categories from TID2013, including:

- additive Gaussian noise
- additive noise in color components
- spatially correlated noise
- masked noise
- high-frequency noise
- impulse noise
- quantization noise
- Gaussian blur
- image denoising
- JPEG compression
- JPEG2000 compression
- JPEG transmission errors
- JPEG2000 transmission errors
- non-eccentricity pattern noise
- local block-wise distortions
- mean shift
- contrast change
- change of color saturation
- multiplicative Gaussian noise
- comfort noise
- lossy compression of color images
- color quantization with dithering
- chromatic aberrations
- sparse sampling and reconstruction

## Repository Structure

```text
Distortion-Classification/
├── README.md
├── .gitignore
├── model.py               # TID2013 dataset loader and transforms
├── train.py               # ConvNeXt-Tiny classifier training
├── predict.py             # Folder-level distortion prediction
├── data.py                # Low-MOS image filtering
├── calculate.py           # Category/statistics calculation
├── classification.py      # Additional classification utilities
├── hebingcsv.py           # CSV merging utility
├── translate.py           # Label/name translation utility
├── 统计图.py               # Plot generation script
├── Koniq10k噪声类别统计图.png
└── SPAQ噪声类别统计图.png
