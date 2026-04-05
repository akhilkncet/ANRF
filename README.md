# Flood Segmentation for ANRFAISEHack Theme 1 Phase 2

This repository contains a notebook-based baseline for pixel-wise flood segmentation on multi-band SAR GeoTIFF imagery.

Notebook included:
- `notebookf34c112244.ipynb`

## Problem Statement

The task is to segment flooded regions at pixel level from 6-channel SAR satellite imagery and generate competition-ready submission masks.

## Approach Used in the Notebook

The notebook follows a supervised semantic-segmentation pipeline using a UNet encoder-decoder model for 3-class pixel prediction, then converts predictions to flood-only masks for submission.

## Methodology

### 1. Environment and Dependency Strategy

- Reinstalls PyTorch with CUDA 12.1 wheels in notebook setup to avoid GPU compatibility issues seen in some Kaggle runtimes.
- Uses `segmentation-models-pytorch` for UNet implementation.
- Uses `imagecodecs` + `tifffile` for reliable multi-band TIFF decoding.

### 2. Data Ingestion and Preprocessing

- Input images are 6-channel SAR GeoTIFF files.
- Custom `FloodDataset` reads image tensors and label masks.
- Image shape is converted from `(H, W, 6)` to `(6, H, W)` for PyTorch channel-first format.
- Label masks are loaded as integer class maps (`int64`) required by `CrossEntropyLoss`.
- Train IDs come from `split/train.txt`; test IDs are discovered by scanning prediction image files.

### 3. Data Loading Policy

- Train loader: batch size 2, shuffle enabled.
- Test loader: batch size 1, deterministic order (no shuffle).
- `num_workers=0` is intentionally used to reduce codec/process issues in notebook environments.
- `pin_memory` is enabled when CUDA is active.

### 4. Model Architecture

- Model: UNet from `segmentation-models-pytorch`.
- Encoder: `resnet18` with ImageNet pretrained weights.
- Input channels: 6.
- Output classes: 3 (multi-class segmentation logits per pixel).

### 5. Training Objective and Optimization

- Loss function: `CrossEntropyLoss` for multi-class segmentation.
- Optimizer: `Adam` with learning rate `1e-4`.
- Epochs: 10.
- Training loop per batch: forward pass -> loss -> backward pass -> optimizer step.
- Tracks average epoch loss and reports best epoch loss.

### 6. Runtime Device Handling

- Primary device: CUDA if available, else CPU.
- Includes runtime fallback: if CUDA kernel/image compatibility errors appear during training, training is restarted on CPU to ensure completion.

### 7. Inference and Post-processing

- Inference uses `model.eval()` and `torch.no_grad()`.
- Per-pixel class is selected using `argmax` on model logits.
- A flood-only binary mask is generated using `FLOOD_CLASS = 1`.
- Binary masks are encoded with Run-Length Encoding (RLE) using Fortran-order flattening to match competition conventions.
- Empty masks are encoded as `0 0`.

### 8. Submission Construction

- Builds a DataFrame with columns: `id`, `rle_mask`.
- Sorts rows by ID for consistent submission ordering.
- Saves final output as `submission.csv`.

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
- RLE encoding uses Fortran-order flattening (`order='F'`) to match expected leaderboard format.
