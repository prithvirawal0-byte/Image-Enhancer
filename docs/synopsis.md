# SEMESTER 3 DEEP LEARNING PROJECT — SYNOPSIS

## Project Title
Image Denoising using Convolutional Autoencoder

## Team Members
- Mohammed Nazil Shaikh
- Prithvi Rawal

## Course
Semester 3 — Deep Learning

## Problem Statement
Image denoising is the task of removing noise (random corruption) from images while preserving important visual features such as edges, textures, and structural details. Real-world images are often degraded by noise due to sensor limitations, low-light conditions, transmission errors, or compression. Traditional filters such as Gaussian and Median filters tend to blur fine details while removing noise.

In this project, we develop a Convolutional Autoencoder that learns to map noisy images to their clean versions. The model is trained on pairs of (noisy input, clean target) images and evaluated using PSNR and SSIM metrics.

## Objectives
1. Build a complete end-to-end deep learning pipeline for image denoising.
2. Train a convolutional autoencoder to reconstruct clean images from noisy inputs.
3. Evaluate performance using PSNR, SSIM, and MSE.
4. Compare against traditional denoising methods (Gaussian, Median, BM3D).
5. Build a working web interface for demonstration.

## Dataset
- Primary: BSD500 (Berkeley Segmentation Dataset) — 500 natural images
- Backup: DIV2K subset or CIFAR-10
- Split: 80% train, 10% validation, 10% test
- Preprocessing: Resize to 256x256, normalize to [0, 1]

## Noise Types
- Gaussian noise (sigma = 15, 25, 50) — primary
- Salt-and-pepper noise — secondary
- Speckle noise — extension

## Model Architecture

Primary: Convolutional Autoencoder
- Encoder: Conv -> ReLU -> Conv -> ReLU -> Conv -> ReLU
- Bottleneck: compressed feature representation
- Decoder: ConvTranspose -> ReLU -> ConvTranspose -> ReLU -> ConvTranspose -> Sigmoid
- Loss: MSE
- Optimizer: Adam (lr = 1e-3)

Backup 1: U-Net (autoencoder with skip connections)
Backup 2: DnCNN (residual noise learning)

## Evaluation Metrics
- PSNR (Peak Signal-to-Noise Ratio) — higher is better
- SSIM (Structural Similarity Index) — closer to 1 is better
- MSE (Mean Squared Error) — lower is better
- Visual comparison: noisy vs denoised vs original

## Baseline Comparisons
- Gaussian Filter
- Median Filter
- BM3D

## Demo Plan

Build a web interface using Streamlit or Gradio where a user can:
- Upload an image
- Choose noise type and level
- View the denoised result
- See PSNR / SSIM metrics
- Compare noisy, denoised, and original side by side

## Backup Plans
- If autoencoder produces blurry results: upgrade to U-Net with skip connections
- If performance is limited: implement DnCNN
- If visual quality is poor but PSNR is good: add SSIM loss
- If GPU is unavailable: train on Google Colab with reduced dataset
- If dataset unavailable: use custom images or CIFAR-10

## Division of Work
- Member A (Mohammed Nazil Shaikh): Dataset collection, noise generation, preprocessing, evaluation metrics, error analysis
- Member B (Prithvi Rawal): Model architecture, training loop, hyperparameter tuning, inference function
- Both: Debugging, results review, UI integration, report, presentation

## Timeline
- Week 1: Synopsis, environment setup, dataset collection
- Week 2: Noise generation pipeline, baseline autoencoder
- Week 3: Training and initial evaluation
- Week 4: Model improvement (U-Net / DnCNN if needed)
- Week 5: Web interface development
- Week 6: Error analysis, final evaluation, report
- Week 7: Final presentation and demo

## Expected Outcome
A trained convolutional autoencoder that removes noise from images while preserving important visual features, evaluated with PSNR and SSIM, and demonstrated through a working web interface.

## References
1. Zhang et al., "Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising" (DnCNN), 2017
2. Mao et al., "Image Restoration Using Very Deep Convolutional Encoder-Decoder Networks with Symmetric Skip Connections" (RED-Net), 2016
3. Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation", 2015
4. BSD500 Dataset: https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/