import cv2
import numpy as np
from PIL import Image


MAX_DIMENSION = 2048


def pil_to_cv2(pil_image: Image.Image) -> np.ndarray:
    """
    Convert PIL RGB image to OpenCV BGR image.
    """

    if pil_image.mode == "RGBA":
        background = Image.new("RGB", pil_image.size, (255, 255, 255))
        background.paste(pil_image, mask=pil_image.split()[3])
        pil_image = background

    elif pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")

    rgb_array = np.array(pil_image)

    bgr_image = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)

    return bgr_image


def cv2_to_pil(cv_image: np.ndarray) -> Image.Image:
    """
    Convert OpenCV BGR image to PIL RGB image.
    """

    rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

    return Image.fromarray(rgb_image)


def resize_if_large(cv_image: np.ndarray) -> np.ndarray:
    """
    Resize very large images to avoid slow processing.
    """

    height, width = cv_image.shape[:2]

    if max(width, height) <= MAX_DIMENSION:
        return cv_image

    scale = MAX_DIMENSION / max(width, height)

    new_width = int(width * scale)
    new_height = int(height * scale)

    resized = cv2.resize(
        cv_image,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA,
    )

    return resized


def estimate_skew_angle(gray: np.ndarray) -> float:
    """
    Estimate document skew angle using foreground pixels.

    This is a lightweight deskew method.
    If the angle cannot be estimated reliably, it returns 0.
    """

    try:
        inverted = cv2.bitwise_not(gray)

        thresholded = cv2.threshold(
            inverted,
            0,
            255,
            cv2.THRESH_BINARY | cv2.THRESH_OTSU,
        )[1]

        coordinates = np.column_stack(np.where(thresholded > 0))

        if len(coordinates) < 100:
            return 0.0

        angle = cv2.minAreaRect(coordinates)[-1]

        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        if abs(angle) > 15:
            return 0.0

        return float(angle)

    except Exception:
        return 0.0


def deskew(cv_image: np.ndarray) -> np.ndarray:
    """
    Deskew image using estimated angle.
    """

    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

    angle = estimate_skew_angle(gray)

    if abs(angle) < 0.1:
        return cv_image

    height, width = cv_image.shape[:2]

    center = (width // 2, height // 2)

    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    rotated = cv2.warpAffine(
        cv_image,
        rotation_matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )

    return rotated


def denoise(cv_image: np.ndarray) -> np.ndarray:
    """
    Reduce small noise while preserving edges.
    """

    return cv2.fastNlMeansDenoisingColored(
        cv_image,
        None,
        h=5,
        hColor=5,
        templateWindowSize=7,
        searchWindowSize=21,
    )


def normalize_brightness(cv_image: np.ndarray) -> np.ndarray:
    """
    Normalize brightness/contrast using CLAHE on the luminance channel.
    """

    lab = cv2.cvtColor(cv_image, cv2.COLOR_BGR2LAB)

    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )

    enhanced_l = clahe.apply(l_channel)

    enhanced_lab = cv2.merge(
        [
            enhanced_l,
            a_channel,
            b_channel,
        ]
    )

    enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    return enhanced_bgr


def preprocess(pil_image: Image.Image) -> Image.Image:
    """
    Full OpenCV preprocessing pipeline:

    1. Convert PIL image to OpenCV format.
    2. Resize if very large.
    3. Deskew document.
    4. Denoise image.
    5. Normalize brightness and contrast.
    6. Convert back to PIL image.
    """

    cv_image = pil_to_cv2(pil_image)

    cv_image = resize_if_large(cv_image)

    cv_image = deskew(cv_image)

    cv_image = denoise(cv_image)

    cv_image = normalize_brightness(cv_image)

    pil_output = cv2_to_pil(cv_image)

    return pil_output