"""
Dataset module for image denoising.

Loads clean images, extracts random patches, normalizes to [0, 1],
and adds Gaussian noise on-the-fly to create (noisy, clean) pairs.
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split


class DenoisingDataset(Dataset):
    """
    PyTorch Dataset for image denoising.

    Each __getitem__ returns:
        noisy_patch: (3, patch_size, patch_size) tensor in [0, 1]
        clean_patch: (3, patch_size, patch_size) tensor in [0, 1]
    """

    def __init__(self, image_dir, patch_size=128, sigma=25, max_images=None):
        self.image_dir = image_dir
        self.patch_size = patch_size
        self.sigma = sigma

        supported = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
        self.image_paths = sorted([
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.lower().endswith(supported)
        ])

        if max_images is not None:
            self.image_paths = self.image_paths[:max_images]

        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {image_dir}")

        print(f"DenoisingDataset: {len(self.image_paths)} images loaded")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        h, w = image.shape[:2]
        ps = self.patch_size

        if h < ps or w < ps:
            scale = max(ps / h, ps / w) + 0.1
            image = cv2.resize(image, (int(w * scale), int(h * scale)))
            h, w = image.shape[:2]

        top = np.random.randint(0, h - ps + 1)
        left = np.random.randint(0, w - ps + 1)
        patch = image[top:top + ps, left:left + ps]

        clean = patch.astype(np.float32) / 255.0

        noise = np.random.normal(0, self.sigma / 255.0, clean.shape).astype(np.float32)
        noisy = np.clip(clean + noise, 0.0, 1.0)

        clean = torch.from_numpy(clean).permute(2, 0, 1).float()
        noisy = torch.from_numpy(noisy).permute(2, 0, 1).float()

        return noisy, clean


def get_dataloaders(clean_dir, patch_size=128, sigma=25, batch_size=8,
                    val_split=0.1, max_images=None, num_workers=0):
    """
    Create train and validation DataLoaders from a folder of clean images.
    """
    full_dataset = DenoisingDataset(
        image_dir=clean_dir,
        patch_size=patch_size,
        sigma=sigma,
        max_images=max_images
    )

    total = len(full_dataset)
    val_size = max(1, int(total * val_split))
    train_size = total - val_size

    train_ds, val_ds = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers
    )

    print(f"Train: {train_size} samples | Val: {val_size} samples")
    return train_loader, val_loader


if __name__ == "__main__":
    loader, _ = get_dataloaders(
        clean_dir="data/clean",
        patch_size=128,
        sigma=25,
        batch_size=2,
        max_images=10
    )

    noisy, clean = next(iter(loader))
    print(f"\nNoisy batch shape: {noisy.shape}")
    print(f"Clean batch shape: {clean.shape}")
    print(f"Noisy range: [{noisy.min():.4f}, {noisy.max():.4f}]")
    print(f"Clean range: [{clean.min():.4f}, {clean.max():.4f}]")
    print("\nDataset loader works.")