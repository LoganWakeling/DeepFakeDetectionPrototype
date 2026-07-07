from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.fftpack import dct


DEFAULT_SIZE = 128


def load_image(image_path: str | Path) -> np.ndarray:
    """Load an image from disk as BGR, matching OpenCV's default format."""
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return image


def to_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        gray = image
    elif image.ndim == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.ndim == 3 and image.shape[2] == 4:
        gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")

    return gray.astype(np.float32)


def compute_dct2(gray_image: np.ndarray, size: int = DEFAULT_SIZE) -> np.ndarray:
    """Resize, normalize, and compute a 2D DCT coefficient matrix."""
    if size <= 0:
        raise ValueError("DCT size must be positive.")

    resized = cv2.resize(gray_image, (size, size), interpolation=cv2.INTER_AREA)
    normalized = resized / 255.0
    normalized = normalized - np.mean(normalized)
    return dct(dct(normalized.T, norm="ortho").T, norm="ortho")


def frequency_energy_bands(dct_coefficients: np.ndarray) -> dict[str, float]:
    """Summarize DCT energy into low, mid, and high frequency triangular bands."""
    coefficients = np.abs(dct_coefficients)
    size = coefficients.shape[0]
    rows, cols = np.indices(coefficients.shape)
    normalized_frequency = (rows + cols) / (2 * (size - 1))

    low_mask = normalized_frequency < 0.15
    mid_mask = (normalized_frequency >= 0.15) & (normalized_frequency < 0.40)
    high_mask = normalized_frequency >= 0.40

    low_energy = float(np.mean(coefficients[low_mask] ** 2))
    mid_energy = float(np.mean(coefficients[mid_mask] ** 2))
    high_energy = float(np.mean(coefficients[high_mask] ** 2))

    return {
        "dct_low_energy": low_energy,
        "dct_mid_energy": mid_energy,
        "dct_high_energy": high_energy,
    }


def extract_frequency_features(
    image: str | Path | np.ndarray,
    size: int = DEFAULT_SIZE,
) -> dict[str, float]:
    """
    Extract frequency-domain features for a full image sample.

    Features match PROJECT_CONTEXT.md:
    - 2D DCT on grayscale image samples
    - low/mid/high frequency energy
    - global std + high/low ratio
    """
    loaded_image = load_image(image) if isinstance(image, (str, Path)) else image
    gray = to_grayscale(loaded_image)
    dct_coefficients = compute_dct2(gray, size=size)

    features = frequency_energy_bands(dct_coefficients)
    high_energy = features["dct_high_energy"]
    low_energy = features["dct_low_energy"]

    features.update(
        {
            "dct_global_std": float(np.std(dct_coefficients)),
            "dct_high_low_ratio": float(high_energy / (low_energy + 1e-8)),
        }
    )
    return features


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract DCT frequency features from a full image sample.")
    parser.add_argument("image_path", type=Path, help="Path to an image sample.")
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE, help="Square size used before DCT.")
    args = parser.parse_args()

    features: dict[str, Any] = extract_frequency_features(args.image_path, size=args.size)
    print(json.dumps(features, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
