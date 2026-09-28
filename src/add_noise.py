"""
Noise generation module for Image Denoising project.

Provides functions to add Gaussian, salt-and-pepper, and speckle noise
to images. Used both for generating evaluation samples and for
on-the-fly augmentation during training.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


# ---------- NOISE FUNCTIONS ----------

def add_gaussian_noise(image, sigma=25):
    """
    Add Gaussian noise to an image.

    Args:
        image: numpy array (H, W) or (H, W, C), values 0-255 (uint8)
        sigma: standard deviation of the noise (15, 25, 50 are standard)

    Returns:
        noisy image, same shape and dtype as input
    """
    noise = np.random.normal(0, sigma, image.shape)
    noisy = image.astype(np.float32) + noise
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy


def add_salt_pepper_noise(image, amount=0.05):
    """
    Add salt-and-pepper noise to an image.

    Args:
        image: numpy array (H, W) or (H, W, C), values 0-255
        amount: fraction of pixels to corrupt (0.05 = 5%)

    Returns:
        noisy image
    """
    noisy = image.copy()
    h, w = image.shape[:2]
    num_pixels = int(amount * h * w)

    # Salt (white pixels)
    coords = [np.random.randint(0, i, num_pixels) for i in (h, w)]
    noisy[coords[0], coords[1]] = 255

    # Pepper (black pixels)
    coords = [np.random.randint(0, i, num_pixels) for i in (h, w)]
    noisy[coords[0], coords[1]] = 0

    return noisy


def add_speckle_noise(image, sigma=0.1):
    """
    Add speckle (multiplicative) noise to an image.

    Args:
        image: numpy array (H, W) or (H, W, C), values 0-255
        sigma: strength of the noise

    Returns:
        noisy image
    """
    noise = np.random.randn(*image.shape) * sigma
    noisy = image.astype(np.float32) + image.astype(np.float32) * noise
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy


# ---------- BATCH PROCESSING ----------

def generate_noisy_images(input_dir, output_dir, noise_type="gaussian",
                          max_images=None, **kwargs):
    """
    Read all images from input_dir, add noise, and save to output_dir.

    Args:
        input_dir: folder with clean images
        output_dir: folder to save noisy images
        noise_type: 'gaussian', 'salt_pepper', or 'speckle'
        max_images: limit number of images to process (None = all)
        **kwargs: passed to noise function (e.g., sigma=25)
    """
    os.makedirs(output_dir, exist_ok=True)

    supported = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    files = sorted([f for f in os.listdir(input_dir)
                    if f.lower().endswith(supported)])

    if not files:
        print(f"No images found in {input_dir}")
        return

    if max_images is not None:
        files = files[:max_images]

    for filename in files:
        path = os.path.join(input_dir, filename)
        image = cv2.imread(path)

        if image is None:
            print(f"Could not read {filename}, skipping.")
            continue

        if noise_type == "gaussian":
            noisy = add_gaussian_noise(image, **kwargs)
        elif noise_type == "salt_pepper":
            noisy = add_salt_pepper_noise(image, **kwargs)
        elif noise_type == "speckle":
            noisy = add_speckle_noise(image, **kwargs)
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")

        out_path = os.path.join(output_dir, filename)
        cv2.imwrite(out_path, noisy)

    print(f"Done. Processed {len(files)} images with {noise_type} noise.")
    print(f"Saved to: {output_dir}")


# ---------- VISUALIZATION ----------

def visualize_comparison(clean_path, noisy_path, title="Clean vs Noisy",
                         save_path="docs/noise_comparison.png"):
    """
    Show clean and noisy images side by side.
    """
    clean = cv2.imread(clean_path)
    noisy = cv2.imread(noisy_path)

    if clean is None or noisy is None:
        print("Could not load images for visualization.")
        return

    clean = cv2.cvtColor(clean, cv2.COLOR_BGR2RGB)
    noisy = cv2.cvtColor(noisy, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(14, 7))

    plt.subplot(1, 2, 1)
    plt.title("Clean (Ground Truth)", fontsize=14)
    plt.imshow(clean)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title(title, fontsize=14)
    plt.imshow(noisy)
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.show()
    print(f"Saved comparison to {save_path}")


# ---------- MAIN ----------

if __name__ == "__main__":
    clean_dir = "data/clean"
    noisy_dir = "data/noisy"

    print("=" * 50)
    print("Generating noisy images...")
    print("=" * 50)

    # Generate Gaussian noise at sigma=25 for 20 images
    print("\nGenerating Gaussian noise (sigma=25)...")
    generate_noisy_images(
        clean_dir, noisy_dir,
        noise_type="gaussian",
        sigma=25,
        max_images=20
    )

        # Visualize one example (skip hidden files like .DS_Store)
    supported = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    samples = sorted([f for f in os.listdir(clean_dir)
                      if f.lower().endswith(supported)])
    if samples:
        sample = samples[0]
        visualize_comparison(
            os.path.join(clean_dir, sample),
            os.path.join(noisy_dir, sample),
            title="Noisy (Gaussian, sigma=25)"
        )
    else:
        print("No images found for visualization.")