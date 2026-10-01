"""
Training script for the convolutional autoencoder denoiser.

Uses a combined MSE + SSIM loss to produce crisper outputs and
reduce the over-smoothing that pure MSE causes.

Usage:
    python src/train.py --epochs 40 --save_path models/autoencoder_ssim.pth
"""

import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from skimage.metrics import structural_similarity

from dataset import get_dataloaders
from model import ConvAutoencoder


# ---------- COMBINED LOSS ----------

class CombinedLoss(nn.Module):
    """
    MSE + SSIM loss.

    alpha = 1.0 -> pure MSE
    alpha = 0.4 -> balanced (recommended)
    alpha = 0.0 -> pure SSIM
    """
    def __init__(self, alpha=0.4):
        super().__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        mse_loss = self.mse(pred, target)

        pred_np = pred.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()
        ssim_vals = []
        for i in range(pred_np.shape[0]):
            p = np.transpose(pred_np[i], (1, 2, 0))
            t = np.transpose(target_np[i], (1, 2, 0))
            ssim_vals.append(
                structural_similarity(t, p, channel_axis=2, data_range=1.0)
            )
        ssim_loss = 1.0 - float(np.mean(ssim_vals))

        return self.alpha * mse_loss + (1 - self.alpha) * ssim_loss


# ---------- METRICS ----------

def compute_psnr(pred, target):
    mse = torch.mean((pred - target) ** 2).item()
    if mse == 0:
        return 100.0
    return 10 * np.log10(1.0 / mse)


def compute_ssim(pred, target):
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()

    scores = []
    for i in range(pred_np.shape[0]):
        p = np.transpose(pred_np[i], (1, 2, 0))
        t = np.transpose(target_np[i], (1, 2, 0))
        score = structural_similarity(t, p, channel_axis=2, data_range=1.0)
        scores.append(score)
    return float(np.mean(scores))


# ---------- TRAIN / VALIDATE ----------

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for noisy, clean in loader:
        noisy = noisy.to(device)
        clean = clean.to(device)

        optimizer.zero_grad()
        output = model(noisy)
        loss = criterion(output, clean)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * noisy.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    n = 0

    for noisy, clean in loader:
        noisy = noisy.to(device)
        clean = clean.to(device)

        output = model(noisy)
        loss = criterion(output, clean)

        total_loss += loss.item() * noisy.size(0)
        total_psnr += compute_psnr(output, clean) * noisy.size(0)
        total_ssim += compute_ssim(output, clean) * noisy.size(0)
        n += noisy.size(0)

    return total_loss / n, total_psnr / n, total_ssim / n


# ---------- MAIN ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean_dir", type=str, default="data/clean")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--patch_size", type=int, default=128)
    parser.add_argument("--sigma", type=float, default=25)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--save_path", type=str, default="models/autoencoder.pth")
    parser.add_argument("--alpha", type=float, default=0.4,
                        help="MSE weight in combined loss (0.4 recommended).")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data
    train_loader, val_loader = get_dataloaders(
        clean_dir=args.clean_dir,
        patch_size=args.patch_size,
        sigma=args.sigma,
        batch_size=args.batch_size,
        max_images=args.max_images,
        num_workers=0
    )

    # Model
    model = ConvAutoencoder().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Loss and optimizer
    criterion = CombinedLoss(alpha=args.alpha)
    optimizer = Adam(model.parameters(), lr=args.lr)

    print(f"Loss: CombinedLoss (alpha={args.alpha} MSE + "
          f"{1 - args.alpha:.1f} SSIM)")

    # Training loop
    best_val_loss = float("inf")
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    print("\n" + "=" * 78)
    print(f"{'Epoch':>6} | {'Train Loss':>10} | {'Val Loss':>10} | "
          f"{'PSNR':>8} | {'SSIM':>8} | {'Time':>6}")
    print("=" * 78)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, psnr, ssim = validate(model, val_loader, criterion, device)

        elapsed = time.time() - t0
        print(f"{epoch:>6} | {train_loss:>10.5f} | {val_loss:>10.5f} | "
              f"{psnr:>7.2f}dB | {ssim:>8.4f} | {elapsed:>5.1f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), args.save_path)

    print("=" * 78)
    print(f"Best validation loss: {best_val_loss:.5f}")
    print(f"Model saved to: {args.save_path}")


if __name__ == "__main__":
    main()