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

In this project, we develop a U-Net-style Convolutional Autoencoder that learns to map noisy images to their clean versions. The model is trained on pairs of (noisy input, clean target) images and evaluated using PSNR and SSIM metrics.

## Objectives
1. Build a complete end-to-end deep learning pipeline for image denoising.
2. Train a U-Net convolutional autoencoder with skip connections to reconstruct clean images from noisy inputs.
3. Use a combined MSE + SSIM loss to reduce over-smoothing.
4. Evaluate performance using PSNR, SSIM, and MSE.
5. Compare against traditional denoising methods (Gaussian, Median filters).
6. Analyze failure cases and document limitations.
7. Build a working web interface for demonstration using Streamlit.

## Dataset
- Primary: DIV2K (DIVerse 2K resolution) — Div2K_Random100 subset (100 high-resolution images)
- Split: 90% train / 10% validation with fixed seed for reproducibility
- Preprocessing: Random 128×128 patch extraction, normalize to [0, 1], on-the-fly Gaussian noise injection
- Alternative datasets considered: CIFAR-10 (rejected — too low resolution), SIDD (attempted for real noise — insufficient data at 40 pairs, documented as future work)

## Noise Types
- Gaussian noise (σ = 25) — primary training noise
- Salt-and-pepper noise — implemented in `add_noise.py`, not trained
- Speckle noise — implemented in `add_noise.py`, not trained

## Model Architecture

Primary: U-Net-style Convolutional Autoencoder with skip connections
- Encoder: 3 ConvBlocks (Conv-BatchNorm-ReLU × 2), downsampling via MaxPool
- Bottleneck: 256 channels at 16×16 resolution
- Decoder: 3 UpBlocks (ConvTranspose + skip concat + ConvBlock)
- Output: 1×1 Conv + Sigmoid → (3, H, W) in [0, 1]
- Skip connections preserve fine details lost in the bottleneck
- Loss: Combined MSE + SSIM (α = 0.4)
- Optimizer: Adam (lr = 1e-3)
- Parameters: 1,928,483

Baselines implemented for comparison:
- Plain Convolutional Autoencoder (no skip connections) — tested, failed
- Gaussian filter (5×5)
- Median filter (5×5)

## Evaluation Metrics
- PSNR (Peak Signal-to-Noise Ratio) — higher is better
- SSIM (Structural Similarity Index) — closer to 1 is better
- MSE (Mean Squared Error) — lower is better
- Visual comparison: noisy vs denoised vs original

## Baseline Comparisons
- Gaussian Filter (5×5 kernel)
- Median Filter (5×5 kernel)
- Noisy input (no denoising) — reference

## Demo Plan

Built a web interface using Streamlit where a user can:
- Upload an image or pick from built-in samples
- Choose input mode: clean image (test with synthetic noise) or already-noisy image
- Select noise level: σ = 15, 25, 50
- View the denoised result with light automatic sharpening
- See PSNR / SSIM metrics (in clean mode)
- Compare noisy, denoised, and original side by side

## Backup Plans

- ✅ *Upgraded plain autoencoder to U-Net with skip connections* — this became the primary model
- ✅ *Added SSIM loss* — combined with MSE to reduce over-smoothing
- ⚠️ *Real-noise training on SIDD* — attempted with 40 pairs, model failed to generalize, documented as limitation
- ❌ *BM3D comparison* — not implemented (computational cost)
- ❌ *Google Colab / GPU training* — not used (CPU only)

## Division of Work
- Mohammed Nazil Shaikh: Dataset collection, noise generation, preprocessing, evaluation metrics, error analysis
- Prithvi Rawal: Model architecture, training loop, hyperparameter tuning, inference function
- Both: Debugging, results review, UI integration, report, presentation

## Timeline
- Week 1: Synopsis, environment setup, dataset collection
- Week 2: Noise generation pipeline, baseline autoencoder
- Week 3: Training and initial evaluation (plain autoencoder failed → switched to U-Net)
- Week 4: Model improvement (U-Net with skip connections, combined MSE+SSIM loss)
- Week 5: Web interface development (Streamlit)
- Week 6: Real-noise experiment (SIDD), error analysis, final evaluation, report
- Week 7: Final presentation and demo

## Expected Outcome
A trained U-Net convolutional autoencoder that removes Gaussian noise from images while preserving important visual features, evaluated with PSNR and SSIM, compared against traditional filters, and demonstrated through a working web interface. The project documents both successful results on synthetic noise and the limitation of real-noise generalization with limited data.

## References
1. Zhang et al., "Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising" (DnCNN), 2017
2. Mao et al., "Image Restoration Using Very Deep Convolutional Encoder-Decoder Networks with Symmetric Skip Connections" (RED-Net), 2016
3. Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation", 2015
4. Agustsson & Timofte, "NTIRE 2017 Challenge on Single Image Super-Resolution: Dataset and Study" (DIV2K), 2017
5. Wang et al., "Image Quality Assessment: From Error Visibility to Structural Similarity" (SSIM), 2004
6. Abdelhamed et al., "A High-Quality Denoising Dataset for Smartphone Cameras" (SIDD), 2018
7. DIV2K Dataset: https://data.vision.ee.ethz.ch/cvl/DIV2K/