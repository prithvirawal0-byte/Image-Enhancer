"""
Baseline comparison for image denoising.

Compares three methods at a given noise level:
    1. Gaussian filter  (traditional)
    2. Median filter    (traditional)
    3. U-Net            (our model)

Outputs:
    - Printed comparison table
    - Bar chart saved to docs/baseline_comparison_sigmaXX.png

Usage:
    python src/baseline.py --sigma 25 --num_images 5
"""

import os
import argparse
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity

from model import UNetDenoiser


# ---------- METRICS ----------

def psnr(pred, target):
    mse = np.mean((pred - target) ** 2)
    if mse == 0:
        return 100.0
    return 10 * np.log10(1.0 / mse)


def ssim(pred, target):
    return structural_similarity(target, pred, channel_axis=2, data_range=1.0)


# ---------- HELPERS ----------

def load_image(path, size=256):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size))
    return img.astype(np.float32) / 255.0


def add_noise(img, sigma):
    noise = np.random.normal(0, sigma / 255.0, img.shape).astype(np.float32)
    return np.clip(img + noise, 0.0, 1.0)


def apply_gaussian_filter(noisy, kernel_size=5):
    """Gaussian blur."""
    return cv2.GaussianBlur(noisy, (kernel_size, kernel_size), 0)


def apply_median_filter(noisy, kernel_size=5):
    """Median filter (needs uint8 for cv2)."""
    noisy_uint8 = (noisy * 255).astype(np.uint8)
    filtered = cv2.medianBlur(noisy_uint8, kernel_size)
    return filtered.astype(np.float32) / 255.0


def unet_denoise(model, noisy, device):
    """Run U-Net on a single image."""
    t = torch.from_numpy(noisy).permute(2, 0, 1).unsqueeze(0).float().to(device)
    with torch.no_grad():
        out = model(t).squeeze(0).permute(1, 2, 0).cpu().numpy()
    return np.clip(out, 0.0, 1.0)


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

    # Load U-Net
    model = UNetDenoiser().to(device)
    if not os.path.exists(args.model_path):
        print(f"ERROR: Model not found at {args.model_path}")
        return
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    print(f"Loaded U-Net from {args.model_path}")

    # Pick test images (last N)
    supported = (".png", ".jpg", ".jpeg")
    files = sorted([f for f in os.listdir(args.clean_dir)
                    if f.lower().endswith(supported)])
    test_files = files[-args.num_images:]

    print(f"\nEvaluating {len(test_files)} images at sigma={args.sigma}")
    print("Comparing: Gaussian filter | Median filter | U-Net\n")

    # Accumulators
    results = {
        "Gaussian":  {"psnr": 0.0, "ssim": 0.0},
        "Median":    {"psnr": 0.0, "ssim": 0.0},
        "U-Net":     {"psnr": 0.0, "ssim": 0.0},
    }

    for i, fname in enumerate(test_files):
        path = os.path.join(args.clean_dir, fname)
        clean = load_image(path, args.image_size)
        noisy = add_noise(clean, args.sigma)

        # Gaussian
        gauss = apply_gaussian_filter(noisy, kernel_size=5)
        # Median
        median = apply_median_filter(noisy, kernel_size=5)
        # U-Net
        unet = unet_denoise(model, noisy, device)

        for name, output in [
            ("Gaussian", gauss),
            ("Median", median),
            ("U-Net", unet),
        ]:
            results[name]["psnr"] += psnr(output, clean)
            results[name]["ssim"] += ssim(output, clean)

        print(f"  [{i+1}/{len(test_files)}] {fname}  "
              f"G:{psnr(gauss, clean):.2f}dB  "
              f"M:{psnr(median, clean):.2f}dB  "
              f"U:{psnr(unet, clean):.2f}dB")

    # Average
    n = len(test_files)
    for name in results:
        results[name]["psnr"] /= n
        results[name]["ssim"] /= n

    # Print table
    print("\n" + "=" * 72)
    print(f"BASELINE COMPARISON (sigma={args.sigma}, n={n} images)")
    print("=" * 72)
    print(f"{'Method':<15} {'PSNR (dB)':>15} {'SSIM':>15}")
    print("-" * 72)

    # Also compute noisy baseline for reference
    # (in case we want to compare to "doing nothing")
    noisy_psnr = 0.0
    noisy_ssim = 0.0
    for fname in test_files:
        path = os.path.join(args.clean_dir, fname)
        clean = load_image(path, args.image_size)
        noisy = add_noise(clean, args.sigma)
        noisy_psnr += psnr(noisy, clean)
        noisy_ssim += ssim(noisy, clean)
    noisy_psnr /= n
    noisy_ssim /= n

    print(f"{'Noisy (no dn)':<15} {noisy_psnr:>15.2f} {noisy_ssim:>15.4f}")
    for name in ["Gaussian", "Median", "U-Net"]:
        print(f"{name:<15} {results[name]['psnr']:>15.2f} "
              f"{results[name]['ssim']:>15.4f}")
    print("=" * 72)

    # ---------- BAR CHART ----------

    os.makedirs("docs", exist_ok=True)

    methods = ["Noisy", "Gaussian", "Median", "U-Net"]
    psnr_vals = [noisy_psnr, results["Gaussian"]["psnr"],
                 results["Median"]["psnr"], results["U-Net"]["psnr"]]
    ssim_vals = [noisy_ssim, results["Gaussian"]["ssim"],
                 results["Median"]["ssim"], results["U-Net"]["ssim"]]

    colors = ["#e74c3c", "#f39c12", "#3498db", "#2ecc71"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.bar(methods, psnr_vals, color=colors)
    ax1.set_ylabel("PSNR (dB)")
    ax1.set_title(f"PSNR comparison at sigma={args.sigma}")
    ax1.grid(axis="y", alpha=0.3)
    for i, v in enumerate(psnr_vals):
        ax1.text(i, v + 0.3, f"{v:.2f}", ha="center", fontsize=11)

    ax2.bar(methods, ssim_vals, color=colors)
    ax2.set_ylabel("SSIM")
    ax2.set_title(f"SSIM comparison at sigma={args.sigma}")
    ax2.set_ylim(0, 1)
    ax2.grid(axis="y", alpha=0.3)
    for i, v in enumerate(ssim_vals):
        ax2.text(i, v + 0.02, f"{v:.4f}", ha="center", fontsize=11)

    plt.tight_layout()
    chart_path = f"docs/baseline_comparison_sigma{int(args.sigma)}.png"
    plt.savefig(chart_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"\nSaved baseline chart: {chart_path}")


if __name__ == "__main__":
    main()