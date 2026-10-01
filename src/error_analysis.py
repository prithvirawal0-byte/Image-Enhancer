"""
Error analysis script for the denoising model.

For selected test images, generates a 4-panel comparison:
    Noisy | Gaussian | U-Net | Clean

Also computes per-image PSNR/SSIM to identify failure cases.

Usage:
    python src/error_analysis.py --sigma 25
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


def gaussian_filter(noisy, k=5):
    return cv2.GaussianBlur(noisy, (k, k), 0)


def unet_denoise(model, noisy, device):
    t = torch.from_numpy(noisy).permute(2, 0, 1).unsqueeze(0).float().to(device)
    with torch.no_grad():
        out = model(t).squeeze(0).permute(1, 2, 0).cpu().numpy()
    return np.clip(out, 0.0, 1.0)


# ---------- MAIN ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean_dir", type=str, default="data/clean")
    parser.add_argument("--model_path", type=str, default="models/autoencoder_ssim.pth")
    parser.add_argument("--sigma", type=float, default=25)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--output", type=str,
                        default="docs/error_analysis.png")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    model = UNetDenoiser().to(device)
    if not os.path.exists(args.model_path):
        print(f"ERROR: Model not found at {args.model_path}")
        return
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    print(f"Loaded U-Net from {args.model_path}")

    # Pick 3 representative cases: best case, worst case, typical case
    # From previous run: 0764 (sunset) = easy, 0785 (coral) = hard, 0762 (building) = typical
    targets = ["0764.png", "0762.png", "0785.png"]

    supported = (".png", ".jpg", ".jpeg")
    all_files = sorted([f for f in os.listdir(args.clean_dir)
                        if f.lower().endswith(supported)])
    available = [f for f in targets if f in all_files]
    if not available:
        # Fallback: use last 3 test images
        available = all_files[-3:]
    print(f"\nAnalyzing {len(available)} images: {available}\n")

    fig, axes = plt.subplots(len(available), 4,
                             figsize=(18, 4.5 * len(available)))
    if len(available) == 1:
        axes = axes.reshape(1, -1)

    for row, fname in enumerate(available):
        path = os.path.join(args.clean_dir, fname)
        clean = load_image(path, args.image_size)
        noisy = add_noise(clean, args.sigma)

        gauss = gaussian_filter(noisy, k=5)
        unet = unet_denoise(model, noisy, device)

        # Print metrics for this image
        print(f"[{fname}]")
        print(f"  Noisy:    PSNR {psnr(noisy, clean):5.2f} dB | SSIM {ssim(noisy, clean):.4f}")
        print(f"  Gaussian: PSNR {psnr(gauss, clean):5.2f} dB | SSIM {ssim(gauss, clean):.4f}")
        print(f"  U-Net:    PSNR {psnr(unet, clean):5.2f} dB | SSIM {ssim(unet, clean):.4f}")
        print()

        # Panels
        panels = [
            (noisy, f"Noisy\nPSNR {psnr(noisy, clean):.2f} dB"),
            (gauss, f"Gaussian\nPSNR {psnr(gauss, clean):.2f} dB"),
            (unet, f"U-Net\nPSNR {psnr(unet, clean):.2f} dB"),
            (clean, "Clean (Ground Truth)"),
        ]

        for col, (img, title) in enumerate(panels):
            axes[row, col].imshow(img)
            axes[row, col].set_title(title, fontsize=12)
            axes[row, col].axis("off")

        # Row label on the left
        axes[row, 0].set_ylabel(fname, fontsize=11)

    plt.tight_layout()
    plt.savefig(args.output, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"Saved error analysis image: {args.output}")


if __name__ == "__main__":
    main()