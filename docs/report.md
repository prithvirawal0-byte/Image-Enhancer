# Image Denoising using Convolutional Autoencoder
## Project Report

**Course:** Semester 3 — Deep Learning  
**Team:** Mohammed Nazil Shaikh, Prithvi Rawal  
**Date started:** 17 September 2026  
**Status:** In Progress

---

## Abstract
*(To be written at the end — 150–200 words summarizing the entire project)*

---

## 1. Introduction

### 1.1 Background
*(What is image denoising? Why is it important? Real-world applications.)*

### 1.2 Motivation
*(Why did we choose this topic?)*

### 1.3 Objectives
*(Bullet list of what we aim to achieve)*

### 1.4 Scope and Limitations
*(What the project covers and what it does not)*

---

## 2. Problem Statement
*(Formal statement of the problem being solved)*

---

## 3. Literature Review

### 3.1 Traditional Denoising Methods
*(Gaussian filter, Median filter, BM3D)*

### 3.2 Deep Learning Approaches
*(Autoencoders, DnCNN, U-Net, RED-Net)*

### 3.3 Comparison and Gap
*(Why our approach is relevant)*

---

## 4. Dataset

### 4.1 Dataset Selection
*(DIV2K — why we chose it, size, resolution, contents)*


### 4.2 Dataset Statistics

| Split | Number of Images | Resolution | Format |
|-------|-----------------|------------|--------|
| Total | 100 | ~2K (2048×1080) | PNG |

**Note:** We are using the Div2K_Random100 subset from Kaggle, which contains 100 high-resolution images randomly sampled from the full DIV2K dataset. All HR images are stored in `data/clean/`. Training will use patch-based extraction with data augmentation, yielding thousands of training samples per epoch.

### 4.3 Preprocessing Pipeline
*(Resize, normalize, train/val/test split)*

### 4.4 Sample Images
*(Visual examples from the dataset)*

---

## 5. Methodology

### 5.1 Noise Generation

We simulate three types of noise to train and evaluate our denoising model.

**Gaussian Noise**

Gaussian noise is additive noise drawn from a normal distribution:

    noisy(x, y) = clean(x, y) + N(0, sigma^2)

Each pixel is independently corrupted with a random value from a normal distribution with mean 0 and standard deviation sigma. We use three standard noise levels:

| Sigma | Noise Level | Description |
|-------|-------------|-------------|
| 15 | Mild | Good quality sensor |
| 25 | Moderate | Standard denoising benchmark |
| 50 | Severe | Low-light or poor sensor |

**Salt-and-Pepper Noise**

A fraction of pixels are randomly set to either 0 (pepper) or 255 (salt). This simulates dead pixels or transmission errors. We use 5% corruption as the standard rate.

**Speckle Noise**

Multiplicative noise where the corruption is proportional to the pixel value:

    noisy(x, y) = clean(x, y) + clean(x, y) * N(0, sigma^2)

This is common in radar and medical imaging.

**Implementation**

The noise generation module is implemented in `src/add_noise.py`. It provides:

- `add_gaussian_noise(image, sigma)` — additive Gaussian noise
- `add_salt_pepper_noise(image, amount)` — impulsive noise
- `add_speckle_noise(image, sigma)` — multiplicative noise
- `generate_noisy_images(...)` — batch processing over a folder
- `visualize_comparison(...)` — side-by-side visualization

For training, noise is added on-the-fly in each batch so the model sees different noise realizations every epoch. This improves generalization. For evaluation, we pre-generate noisy versions of the test set with fixed random seeds so results are reproducible.

A sample comparison of a clean DIV2K image and its Gaussian-noisy version (sigma=25) is shown below:

![Noise comparison](noise_comparison.png)

The visual degradation is clearly visible, with grain appearing across flat regions and edges becoming less defined.

### 5.2 Model Architecture

We implement a convolutional autoencoder with an encoder–decoder structure.

**Encoder**

The encoder progressively compresses the input image using convolutional blocks:

| Layer | Type | Channels | Output Size (for 128×128 input) |
|-------|------|----------|--------------------------------|
| Block 1 | Conv 3×3 + ReLU × 2 + MaxPool | 3 → 32 | 64 × 64 |
| Block 2 | Conv 3×3 + ReLU × 2 + MaxPool | 32 → 64 | 32 × 32 |
| Block 3 | Conv 3×3 + ReLU × 2 | 64 → 128 | 32 × 32 |

The encoder learns a compact feature representation that captures the structural content of the image while suppressing noise.

**Decoder**

The decoder reconstructs the image using transposed convolutions:

| Layer | Type | Channels | Output Size |
|-------|------|----------|-------------|
| Block 3 reverse | Conv 3×3 + ReLU + ConvTranspose | 128 → 64 | 64 × 64 |
| Block 2 reverse | Conv 3×3 + ReLU + ConvTranspose | 64 → 32 | 128 × 128 |
| Block 1 reverse | Conv 3×3 + ReLU + Conv 3×3 + Sigmoid | 32 → 3 | 128 × 128 |

The final Sigmoid activation constrains the output to [0, 1], matching the normalized input range.

**Why an autoencoder?**

Autoencoders are well suited for denoising because:
- The bottleneck forces the model to learn a compact representation
- Noise is random and hard to encode compactly, so it gets discarded
- The decoder reconstructs clean structure from the compressed representation

**Model size:** 522,691 trainable parameters.

**Input/Output:** Both are (3, H, W) tensors normalized to [0, 1]. The output is the model's estimate of the clean image.
### 5.3 Training Procedure

**Data Pipeline**

Training uses a custom PyTorch Dataset (`src/dataset.py`) that:

1. Loads a clean image from `data/clean/`
2. Randomly crops a 128×128 patch
3. Normalizes pixel values from [0, 255] to [0, 1]
4. Adds Gaussian noise with a given sigma to create the noisy input
5. Returns (noisy_patch, clean_patch) as tensors of shape (3, 128, 128)

Because patches are cropped randomly at every epoch, the model sees a different training sample each time. This acts as data augmentation and effectively multiplies the dataset size.

**Train / Validation Split**

The dataset is split 90% train / 10% validation using a fixed random seed for reproducibility.

**On-the-fly Noise**

Noise is added inside `__getitem__`, so:
- Each epoch sees different noise realizations
- No pre-generated noisy images are needed on disk
- The same pipeline can produce any noise level by changing `sigma`

**Loss Function**

Mean Squared Error (MSE):

    L = (1/N) * sum( (clean - denoised)^2 )

MSE penalizes pixel-level differences and is the standard loss for denoising.

**Optimizer**

Adam with learning rate 1e-3. Adam adapts the learning rate per parameter, which works well for image reconstruction tasks without extensive tuning.

**Batch Size**

8 patches per batch. This fits comfortably in CPU memory for 128×128×3 tensors.

**Epochs**

20–50 depending on available time. On CPU, each epoch takes 1–3 minutes on 100 images with patch_size=128 and batch_size=8.

**Hardware**

Apple MacBook Air (CPU only). Training on a GPU is not required for this project but would speed up iteration significantly.

### 5.4 Evaluation Metrics
*(PSNR, SSIM, MSE — formulas and meaning)*

### 5.5 Baseline Methods
*(Traditional filters used for comparison)*

---

## 6. Experiments and Results

### 6.1 Experimental Setup
*(Hardware, software, hyperparameters)*

### 6.2 Training Curves
*(Loss vs epoch plots)*

### 6.3 Quantitative Results
*(PSNR, SSIM, MSE table across noise levels)*

### 6.4 Comparison with Baselines
*(Table: our model vs Gaussian vs Median vs BM3D)*

### 6.5 Visual Results
*(Side-by-side: noisy vs denoised vs original)*

### 6.6 Ablation Studies
*(What happens if we change sigma, loss function, architecture)*

---

## 7. Error Analysis

### 7.1 Cases Where Model Failed
*(Difficult images, high noise, textures)*

### 7.2 Types of Errors
*(Blurring, artifacts, lost details)*

### 7.3 Improvements Attempted
*(U-Net upgrade, SSIM loss, etc.)*

---

## 8. Demo / User Interface

### 8.1 Design
*(Streamlit/Gradio layout)*

### 8.2 Features
*(Upload, noise controls, metrics, comparison)*

### 8.3 Screenshots
*(UI screenshots)*

---

## 9. Conclusion

### 9.1 Summary of Work
### 9.2 Key Findings
### 9.3 Limitations
### 9.4 Future Work

---

## 10. References
*(To be filled — IEEE format)*

1. Zhang et al., "Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising" (DnCNN), 2017
2. Mao et al., "Image Restoration Using Very Deep Convolutional Encoder-Decoder Networks with Symmetric Skip Connections" (RED-Net), 2016
3. Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation", 2015
4. DIV2K Dataset: https://data.vision.ee.ethz.ch/cvl/DIV2K/

---

## Appendix

### A. Code Structure
*(To be filled)*

### B. Training Logs
*(To be filled)*

### C. Additional Results
*(To be filled)*