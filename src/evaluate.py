"""
Evaluation script for the trained denoising model.

Loads the trained autoencoder, denoises test images at multiple noise
levels, computes metrics, and produces visual comparisons.

Usage:
    python src/evaluate.py
    python src/evaluate.py --num_images 5 --sigma 25
"""

import os
import argparse
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity

from model import ConvAutoencoder
from dataset import DenoisingDataset


# ---------- METRICS ----------

def psnr(pred, target):
    mse = np.mean((pred - target) ** 2)
    if mse == 0:
        return 100.0
    return 10 * np.log10(1.0 / mse)


def ssim(pred, target):
    return structural_similarity(target, pred, channel_axis=2, data_range=1.0)


# ---------- HELPERS ----------

def tensor_to_np(t):
    """(3, H, W) tensor in [0,1] -> (H, W, 3) numpy array."""
    return t.detach().cpu().permute(1, 2, 0).numpy()


def load_image(path, size=256):
    """Load, resize, normalize a clean image."""
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size))
    return img.astype(np.float32) / 255.0


def add_noise(img, sigma):
    """Add Gaussian noise in [0,1] range."""
    noise = np.random.normal(0, sigma / 255.0, img.shape).astype(np.float32)
    return np.clip(img + noise, 0.0, 1.0)


def to_tensor(img):
    """(H, W, 3) numpy in [0,1] -> (1, 3, H, W) tensor."""
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()


# ---------- MAIN ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean_dir", type=str, default="data/clean")
    parser.add_argument("--model_path", type=str, default="models/autoencoder.pth")
    parser.add_argument("--num_images", type=int, default=5)
    parser.add_argument("--sigma", type=float, default=25)
    parser.add_argument("--image_size", type=int, default=256)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    model = ConvAutoencoder().to(device)
    if not os.path.exists(args.model_path):
        print(f"ERROR: Model file not found at {args.model_path}")
        print("Train the model first: python src/train.py")
        return

    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    print(f"Loaded model from {args.model_path}")

    # Get test images (last N, not used in training ideally)
    supported = (".png", ".jpg", ".jpeg")
    all_files = sorted([
        f for f in os.listdir(args.clean_dir)
        if f.lower().endswith(supported)
    ])
    test_files = all_files[-args.num_images:]

    print(f"\nEvaluating on {len(test_files)} images at sigma={args.sigma}")

    # Metrics accumulators
    total_psnr_noisy = 0.0
    total_psnr_denoised = 0.0
    total_ssim_noisy = 0.0
    total_ssim_denoised = 0.0

    # Store for visualization
    samples = []

    with torch.no_grad():
        for i, fname in enumerate(test_files):
            path = os.path.join(args.clean_dir, fname)

            clean = load_image(path, size=args.image_size)
            noisy = add_noise(clean, args.sigma)

            noisy_t = to_tensor(noisy).to(device)
            denoised_t = model(noisy_t).squeeze(0)
            denoised = tensor_to_np(denoised_t)

            # Metrics
            p_noisy = psnr(noisy, clean)
            p_denoised = psnr(denoised, clean)
            s_noisy = ssim(noisy, clean)
            s_denoised = ssim(denoised, clean)

            total_psnr_noisy += p_noisy
            total_psnr_denoised += p_denoised
            total_ssim_noisy += s_noisy
            total_ssim_denoised += s_denoised

            print(f"  [{i+1}/{len(test_files)}] {fname} | "
                  f"Noisy PSNR: {p_noisy:.2f} dB | "
                  f"Denoised PSNR: {p_denoised:.2f} dB")

            samples.append({
                "filename": fname,
                "clean": clean,
                "noisy": noisy,
                "denoised": denoised,
                "psnr_noisy": p_noisy,
                "psnr_denoised": p_denoised,
                "ssim_noisy": s_noisy,
                "ssim_denoised": s_denoised,
            })

    n = len(samples)
    avg_psnr_noisy = total_psnr_noisy / n
    avg_psnr_denoised = total_psnr_denoised / n
    avg_ssim_noisy = total_ssim_noisy / n
    avg_ssim_denoised = total_ssim_denoised / n

    # Print summary
    print("\n" + "=" * 70)
    print(f"SUMMARY (sigma={args.sigma}, n={n} images)")
    print("=" * 70)
    print(f"{'Metric':<20} {'Noisy':>12} {'Denoised':>12} {'Improvement':>15}")
    print("-" * 70)
    print(f"{'PSNR (dB)':<20} {avg_psnr_noisy:>12.2f} "
          f"{avg_psnr_denoised:>12.2f} "
          f"{avg_psnr_denoised - avg_psnr_noisy:>+15.2f}")
    print(f"{'SSIM':<20} {avg_ssim_noisy:>12.4f} "
          f"{avg_ssim_denoised:>12.4f} "
          f"{avg_ssim_denoised - avg_ssim_noisy:>+15.4f}")
    print("=" * 70)

    # ---------- VISUAL COMPARISON ----------

    os.makedirs("docs", exist_ok=True)

    fig, axes = plt.subplots(n, 3, figsize=(15, 5 * n))
    if n == 1:
        axes = axes.reshape(1, -1)

    for i, s in enumerate(samples):
        axes[i, 0].imshow(s["noisy"])
        axes[i, 0].set_title(f"Noisy\nPSNR {s['psnr_noisy']:.2f} dB",
                             fontsize=12)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(s["denoised"])
        axes[i, 1].set_title(f"Denoised\nPSNR {s['psnr_denoised']:.2f} dB",
                             fontsize=12)
        axes[i, 1].axis("off")

        axes[i, 2].imshow(s["clean"])
        axes[i, 2].set_title("Clean (Ground Truth)", fontsize=12)
        axes[i, 2].axis("off")

    plt.tight_layout()
    comparison_path = f"docs/evaluation_comparison_sigma{int(args.sigma)}.png"
    plt.savefig(comparison_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"\nSaved visual comparison: {comparison_path}")

    # ---------- BAR CHART ----------

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    labels = ["Noisy", "Denoised"]
    psnr_vals = [avg_psnr_noisy, avg_psnr_denoised]
    ssim_vals = [avg_ssim_noisy, avg_ssim_denoised]

    colors = ["#e74c3c", "#2ecc71"]

    ax1.bar(labels, psnr_vals, color=colors)
    ax1.set_ylabel("PSNR (dB)")
    ax1.set_title(f"PSNR at sigma={args.sigma}")
    ax1.grid(axis="y", alpha=0.3)
    for i, v in enumerate(psnr_vals):
        ax1.text(i, v + 0.3, f"{v:.2f}", ha="center", fontsize=11)

    ax2.bar(labels, ssim_vals, color=colors)
    ax2.set_ylabel("SSIM")
    ax2.set_title(f"SSIM at sigma={args.sigma}")
    ax2.set_ylim(0, 1)
    ax2.grid(axis="y", alpha=0.3)
    for i, v in enumerate(ssim_vals):
        ax2.text(i, v + 0.02, f"{v:.4f}", ha="center", fontsize=11)

    plt.tight_layout()
    chart_path = f"docs/evaluation_chart_sigma{int(args.sigma)}.png"
    plt.savefig(chart_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"Saved metric chart:   {chart_path}")


if __name__ == "__main__":
    main()
    
