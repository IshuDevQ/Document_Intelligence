import numpy as np
from PIL import Image, ImageFilter

from app.config import (
    IMAGE_RESIZE,
    BRIGHT_THRESHOLD,
    HIGH_THRESHOLD,
    BLOCK_SIZE,
)


FEATURE_NAMES = [
    "ela_mean",
    "ela_std",
    "ela_max",
    "ela_p90",
    "ela_p95",
    "ela_bright_pct",
    "ela_high_pct",
    "ela_region_std",
    "noise_var",
    "edge_density",
    "texture_contrast",
    "texture_energy",
    "color_std_r",
    "color_std_g",
    "aspect_ratio",
    "brightness_mean",
    "brightness_std",
    "ocr_confidence",
]


def extract_features(
    ela_gray: np.ndarray,
    original_pil: Image.Image,
    ocr_confidence: float = 0.5,
) -> np.ndarray:
    features = []

    ela_flat = ela_gray.flatten()

    features.append(float(np.mean(ela_flat)))
    features.append(float(np.std(ela_flat)))
    features.append(float(np.max(ela_flat)))
    features.append(float(np.percentile(ela_flat, 90)))
    features.append(float(np.percentile(ela_flat, 95)))
    features.append(float(np.mean(ela_flat > BRIGHT_THRESHOLD)))
    features.append(float(np.mean(ela_flat > HIGH_THRESHOLD)))

    height, width = ela_gray.shape
    block_means = []

    for y in range(0, height - BLOCK_SIZE + 1, BLOCK_SIZE):
        for x in range(0, width - BLOCK_SIZE + 1, BLOCK_SIZE):
            block = ela_gray[y:y + BLOCK_SIZE, x:x + BLOCK_SIZE]
            block_means.append(np.mean(block))

    if block_means:
        features.append(float(np.std(block_means)))
    else:
        features.append(0.0)

    gray_image = original_pil.convert("L").resize(IMAGE_RESIZE)
    gray_np = np.array(gray_image, dtype=np.float32)

    blurred_np = np.array(
        gray_image.filter(ImageFilter.GaussianBlur(2)),
        dtype=np.float32,
    )

    noise = gray_np - blurred_np
    features.append(float(np.var(noise)))

    gy = np.abs(np.diff(gray_np, axis=0, prepend=gray_np[:1]))
    gx = np.abs(np.diff(gray_np, axis=1, prepend=gray_np[:, :1]))

    gradient_magnitude = np.sqrt(gx ** 2 + gy ** 2)

    edge_threshold = np.mean(gradient_magnitude) + np.std(gradient_magnitude)

    features.append(float(np.mean(gradient_magnitude > edge_threshold)))

    local_mean = np.array(
        gray_image.filter(ImageFilter.BoxBlur(3)),
        dtype=np.float32,
    )

    features.append(float(np.std(gray_np - local_mean)))

    features.append(float(np.mean((gray_np / 255.0) ** 2)))

    rgb_np = np.array(
        original_pil.convert("RGB").resize(IMAGE_RESIZE),
        dtype=np.float32,
    )

    features.append(float(np.std(rgb_np[:, :, 0])))
    features.append(float(np.std(rgb_np[:, :, 1])))

    width_original, height_original = original_pil.size

    if height_original > 0:
        aspect_ratio = float(width_original / height_original)
    else:
        aspect_ratio = 1.0

    features.append(aspect_ratio)

    features.append(float(np.mean(gray_np)))
    features.append(float(np.std(gray_np)))

    features.append(float(np.clip(ocr_confidence, 0.0, 1.0)))

    return np.array(features, dtype=np.float32)