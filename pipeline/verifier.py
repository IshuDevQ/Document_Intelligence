import io
import pickle
import time
import uuid
from dataclasses import dataclass

import numpy as np
from PIL import Image

from app.config import (
    CLASSIFIER_PATH,
    LABEL_NAMES,
    CONFIDENCE_THRESHOLD,
    RESULTS_DIR,
)

from pipeline.preprocessor import preprocess
from pipeline.ela_analyzer import compute_ela_from_pil
from pipeline.feature_extractor import extract_features


if CLASSIFIER_PATH.exists():
    with open(CLASSIFIER_PATH, "rb") as file:
        MODEL_BUNDLE = pickle.load(file)

    CLASSIFIER = MODEL_BUNDLE["classifier"]

else:
    MODEL_BUNDLE = None
    CLASSIFIER = None


@dataclass
class VerificationResult:
    request_id: str
    label: str
    confidence: float
    authentic_prob: float
    tampered_prob: float
    is_uncertain: bool
    ela_mean: float
    ela_region_std: float
    ela_image_path: str
    ocr_text: str
    ocr_confidence: float
    processing_time_ms: int


def save_ela_image(ela_rgb: np.ndarray, request_id: str) -> str:
    """
    Save ELA map image and return relative path.
    """

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    ela_image = Image.fromarray(ela_rgb)

    filename = f"ela_{request_id}.png"

    output_path = RESULTS_DIR / filename

    ela_image.save(output_path)

    return str(output_path)


def run_ocr(pil_image: Image.Image) -> tuple[str, float]:
    try:
        import pytesseract

        data = pytesseract.image_to_data(
            pil_image,
            output_type=pytesseract.Output.DICT,
        )

        words = []
        confidences = []

        for text, confidence in zip(data["text"], data["conf"]):
            text = text.strip()

            if not text:
                continue

            try:
                confidence_value = float(confidence)
            except ValueError:
                continue

            words.append(text)

            if confidence_value > 0:
                confidences.append(confidence_value)

        extracted_text = " ".join(words)

        if confidences:
            ocr_confidence = float(np.mean(confidences)) / 100.0
        else:
            ocr_confidence = 0.5

        return extracted_text, ocr_confidence

    except Exception:
        return "", 0.5


def verify_document(file_bytes: bytes) -> VerificationResult:
    if CLASSIFIER is None:
        raise RuntimeError(
            "Model not found. First run: python -m app.training.train"
        )

    start_time = time.time()

    request_id = str(uuid.uuid4())[:8]

    image = Image.open(io.BytesIO(file_bytes))
    image = preprocess(image)

    ela_rgb, ela_gray = compute_ela_from_pil(image)

    ela_image_path = save_ela_image(
        ela_rgb=ela_rgb,
        request_id=request_id,
    )

    ocr_text, ocr_confidence = run_ocr(image)

    feature_vector = extract_features(
        ela_gray=ela_gray,
        original_pil=image,
        ocr_confidence=ocr_confidence,
    )

    X = feature_vector.reshape(1, -1)

    probabilities = CLASSIFIER.predict_proba(X)[0]

    authentic_prob = float(probabilities[0])
    tampered_prob = float(probabilities[1])

    predicted_label_id = int(CLASSIFIER.predict(X)[0])
    label = LABEL_NAMES[predicted_label_id]

    confidence = float(max(probabilities))

    is_uncertain = confidence < CONFIDENCE_THRESHOLD

    processing_time_ms = int((time.time() - start_time) * 1000)

    return VerificationResult(
        request_id=request_id,
        label=label,
        confidence=round(confidence, 4),
        authentic_prob=round(authentic_prob, 4),
        tampered_prob=round(tampered_prob, 4),
        is_uncertain=is_uncertain,
        ela_mean=round(float(feature_vector[0]), 4),
        ela_region_std=round(float(feature_vector[7]), 4),
        ela_image_path=ela_image_path,
        ocr_text=ocr_text[:1000],
        ocr_confidence=round(float(ocr_confidence), 4),
        processing_time_ms=processing_time_ms,
    )