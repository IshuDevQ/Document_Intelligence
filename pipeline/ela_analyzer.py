import io

import numpy as np
from PIL import Image

from app.config import ELA_QUALITY, ELA_AMPLIFY, IMAGE_RESIZE


def compute_ela_from_pil(pil_image: Image.Image) -> tuple[np.ndarray, np.ndarray]:
    original = pil_image.convert("RGB").resize(IMAGE_RESIZE, Image.LANCZOS)

    buffer = io.BytesIO()
    original.save(buffer, format="JPEG", quality=ELA_QUALITY)
    buffer.seek(0)

    recompressed = Image.open(buffer).convert("RGB")

    original_array = np.array(original, dtype=np.float32)
    recompressed_array = np.array(recompressed, dtype=np.float32)

    difference = np.abs(original_array - recompressed_array)

    ela_rgb = np.clip(difference * ELA_AMPLIFY, 0, 255).astype(np.uint8)
    ela_gray = np.mean(ela_rgb, axis=2).astype(np.float32)

    return ela_rgb, ela_gray