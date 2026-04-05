# Flood Segmentation for ANRFAISEHack Theme 1 Phase 2

This repository contains a notebook-based baseline for pixel-wise flood segmentation on multi-band SAR GeoTIFF imagery.

Notebook included:
- `notebookf34c112244.ipynb`

## Problem Statement

The task is to segment flooded regions at pixel level from 6-channel SAR satellite imagery and generate competition-ready submission masks.

## End-to-End Pipeline

1. Load 6-band SAR GeoTIFF images from competition data folders.
2. Build a custom PyTorch `Dataset` (`FloodDataset`) for train/test splits.
3. Train a UNet segmentation model with a ResNet-18 encoder.
4. Predict per-pixel classes on test images.
5. Convert flood-class predictions to binary masks.
6. Encode masks using Run-Length Encoding (RLE).
7. Export `submission.csv` with columns: `id`, `rle_mask`.

## Model Configuration (from notebook)

- Architecture: UNet (`segmentation_models_pytorch`)
- Encoder: `resnet18` with ImageNet pretrained weights
- Input channels: 6
- Output classes: 3
- Loss: `CrossEntropyLoss`
- Optimizer: `Adam` with learning rate `1e-4`
- Epochs: 10
- Train batch size: 2
- Test batch size: 1

## Dataset Layout Expected in Notebook

The notebook expects Kaggle-style paths:

- `/kaggle/input/competitions/anrfaisehack-theme-1-phase2/data/image`
- `/kaggle/input/competitions/anrfaisehack-theme-1-phase2/data/label`
- `/kaggle/input/competitions/anrfaisehack-theme-1-phase2/data/prediction/image`
- `/kaggle/input/competitions/anrfaisehack-theme-1-phase2/data/split/train.txt`

If your data location differs, update `DATA_PATH` in the notebook.

## Key Notebook Sections

- Dependency setup (PyTorch + CUDA wheels, segmentation libraries)
- Imports and device selection (CUDA with CPU fallback)
- Path setup and sample ID loading
- `FloodDataset` implementation for TIFF images and masks
- DataLoader creation
- UNet model creation
- Loss and optimizer setup
- Training loop with runtime CUDA fallback handling
- RLE encoder implementation
- Inference and flood mask extraction
- Submission CSV generation

## Output

After running the final cells, the notebook writes:
- `submission.csv`

The CSV contains one row per test sample and is ready for competition submission.

## Notes

- The notebook assumes flood class index is `1` (`FLOOD_CLASS = 1`). Verify this against competition label definitions before submission.
- `num_workers=0` is used to avoid environment-specific TIFF codec loader issues.
