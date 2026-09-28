"""
U-Net style convolutional autoencoder for image denoising.

Unlike a plain autoencoder, this uses skip connections between the
encoder and decoder to preserve fine details, color, and edges.

Input:  noisy image (3, H, W), values in [0, 1]
Output: denoised image (3, H, W), values in [0, 1]
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Two 3x3 convs with ReLU and BatchNorm."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    """Upsample + conv, then concatenate with skip, then ConvBlock."""
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
        self.conv = ConvBlock(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNetDenoiser(nn.Module):
    def __init__(self, in_channels=3, base_channels=32):
        super().__init__()

        # ---------- ENCODER ----------
        self.enc1 = ConvBlock(in_channels, base_channels)         # 128x128 x 32
        self.enc2 = ConvBlock(base_channels, base_channels * 2)   # 64x64  x 64
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4)  # 32x32 x 128

        self.pool = nn.MaxPool2d(2)

        # ---------- BOTTLENECK ----------
        self.bottleneck = ConvBlock(base_channels * 4, base_channels * 8)  # 16x16 x 256

        # ---------- DECODER ----------
        self.up3 = UpBlock(base_channels * 8, base_channels * 4, base_channels * 4)
        self.up2 = UpBlock(base_channels * 4, base_channels * 2, base_channels * 2)
        self.up1 = UpBlock(base_channels * 2, base_channels, base_channels)

        # ---------- OUTPUT ----------
        self.out = nn.Conv2d(base_channels, in_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)              # 128x128 x 32
        e2 = self.enc2(self.pool(e1))  # 64x64  x 64
        e3 = self.enc3(self.pool(e2))  # 32x32  x 128

        # Bottleneck
        b = self.bottleneck(self.pool(e3))  # 16x16 x 256

        # Decoder with skips
        d3 = self.up3(b, e3)   # 32x32 x 128
        d2 = self.up2(d3, e2)  # 64x64 x 64
        d1 = self.up1(d2, e1)  # 128x128 x 32

        out = self.out(d1)
        return self.sigmoid(out)


# Backward-compatible alias so existing imports keep working
ConvAutoencoder = UNetDenoiser


if __name__ == "__main__":
    model = UNetDenoiser()
    x = torch.randn(2, 3, 128, 128)
    y = model(x)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Total parameters: {total_params:,}")