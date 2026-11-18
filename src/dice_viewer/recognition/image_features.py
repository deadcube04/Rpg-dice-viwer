"""Versioned scene descriptors from a bounded copy of the decoded RGB image."""

import numpy as np
from PIL import Image

FEATURE_VERSION = "rgb256-bilinear-bt601-lap4-v1"
MAX_SAMPLE_SIDE = 256


def extract_image_features(image):
    """Return JSON-safe numbers only. Never mutate, reorient or redecode image."""
    if image.mode != "RGB" or image.width < 1 or image.height < 1:
        raise ValueError("Expected a nonempty decoded RGB image")
    width, height = image.size
    scale = min(1.0, MAX_SAMPLE_SIDE / max(width, height))
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    # Only the bounded sample becomes float64; the model retains original RGB.
    with image.resize(size, resample=Image.Resampling.BILINEAR) as sample:
        rgb = np.asarray(sample, dtype=np.float64) / 255.0
    channel_means = rgb.mean(axis=(0, 1))
    channel_stds = rgb.std(axis=(0, 1), ddof=0)
    luma = rgb[:, :, 0] * 0.299 + rgb[:, :, 1] * 0.587 + rgb[:, :, 2] * 0.114
    laplacian_variance, valid_pixels = None, 0
    if min(size) >= 3:
        laplacian = (4 * luma[1:-1, 1:-1] - luma[:-2, 1:-1] - luma[2:, 1:-1]
                     - luma[1:-1, :-2] - luma[1:-1, 2:])
        laplacian_variance = float(laplacian.var(ddof=0))
        valid_pixels = int(laplacian.size)
    channels = ("red", "green", "blue")
    return {
        "width": width, "height": height, "pixels": width * height,
        "aspect_ratio": width / height,
        "sample_width": size[0], "sample_height": size[1],
        "rgb_mean": dict(zip(channels, map(float, channel_means))),
        "rgb_std": dict(zip(channels, map(float, channel_stds))),
        "brightness_mean": float(luma.mean()), "contrast_std": float(luma.std(ddof=0)),
        "laplacian_variance": laplacian_variance,
        "laplacian_valid_pixels": valid_pixels,
    }
