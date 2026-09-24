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
*(Number of images, resolution distribution)*

### 4.3 Preprocessing Pipeline
*(Resize, normalize, train/val/test split)*

### 4.4 Sample Images
*(Visual examples from the dataset)*

---

## 5. Methodology

### 5.1 Noise Generation
*(Gaussian, salt-and-pepper, speckle — formulas, sigma values)*

### 5.2 Model Architecture
*(Convolutional Autoencoder — layer by layer description)*

### 5.3 Training Procedure
*(Loss function, optimizer, batch size, epochs, hardware)*

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