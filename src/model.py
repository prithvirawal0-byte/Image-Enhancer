"""
Convolutional Autoencoder for image denoising.

Encoder compresses the noisy image into a compact feature representation.
Decoder reconstructs a clean image from that representation.

Input:  noisy image (3, H, W), values in [0, 1]
Output: denoised image (3, H, W), values in [0, 1]
"""

import torch
import torch.nn as nn


class ConvAutoencoder(nn.Module):
    def __init__(self, in_channels=3, base_channels=32):
        super().__init__()

        # ---------- ENCODER ----------
        self.encoder = nn.Sequential(
            # Block 1: 3 -> 32
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # H/2, W/2

            # Block 2: 32 -> 64
            nn.Conv2d(base_channels, base_channels * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # H/4, W/4

            # Block 3: 64 -> 128
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 4, base_channels * 4, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        # ---------- DECODER ----------
        self.decoder = nn.Sequential(
            # Block 3 reverse
            nn.Conv2d(base_channels * 4, base_channels * 4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2),
            nn.ReLU(inplace=True),

            # Block 2 reverse
            nn.Conv2d(base_channels * 2, base_channels * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2),
            nn.ReLU(inplace=True),

            # Block 1 reverse
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, in_channels, 3, padding=1),
            nn.Sigmoid()  # Output in [0, 1]
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


if __name__ == "__main__":
    # Quick test
    model = ConvAutoencoder()
    x = torch.randn(2, 3, 128, 128)
    y = model(x)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Total parameters: {total_params:,}")