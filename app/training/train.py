from pathlib import Path
import pickle

import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from app.config import (
    AUTHENTIC_DIR,
    TAMPERED_DIR,
    CLASSIFIER_PATH,
    LABEL_AUTHENTIC,
    LABEL_TAMPERED,
)

from pipeline.preprocessor import preprocess
from pipeline.ela_analyzer import compute_ela_from_pil
from pipeline.feature_extractor import extract_features, FEATURE_NAMES
from database.db import save_model_performance


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}


def list_images(folder: Path) -> list[Path]:
    images = []

    for path in folder.rglob("*"):
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(path)

    return images


def extract_single_image_features(image_path: Path, label: int):
    try:
        image = Image.open(image_path)
        image = preprocess(image)

        _, ela_gray = compute_ela_from_pil(image)

        features = extract_features(
            ela_gray=ela_gray,
            original_pil=image,
            ocr_confidence=0.5,
        )

        return features, label

    except Exception as error:
        print(f"Skipping {image_path}: {error}")
        return None


def build_dataset():
    authentic_images = list_images(AUTHENTIC_DIR)
    tampered_images = list_images(TAMPERED_DIR)

    print("Authentic images found:", len(authentic_images))
    print("Tampered images found:", len(tampered_images))

    if len(authentic_images) == 0:
        raise RuntimeError(f"No authentic images found in {AUTHENTIC_DIR}")

    if len(tampered_images) == 0:
        raise RuntimeError(f"No tampered images found in {TAMPERED_DIR}")

    X = []
    y = []

    print()
    print("Extracting authentic image features...")

    for index, image_path in enumerate(authentic_images, start=1):
        result = extract_single_image_features(image_path, LABEL_AUTHENTIC)

        if result is not None:
            features, label = result
            X.append(features)
            y.append(label)

        if index % 500 == 0:
            print(f"Processed authentic: {index}/{len(authentic_images)}")

    print()
    print("Extracting tampered image features...")

    for index, image_path in enumerate(tampered_images, start=1):
        result = extract_single_image_features(image_path, LABEL_TAMPERED)

        if result is not None:
            features, label = result
            X.append(features)
            y.append(label)

        if index % 500 == 0:
            print(f"Processed tampered: {index}/{len(tampered_images)}")

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    return X, y


def train():
    X, y = build_dataset()

    print()
    print("Feature matrix shape:", X.shape)
    print("Label vector shape:", y.shape)
    print("Number of features:", len(FEATURE_NAMES))

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    classifier = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    print()
    print("Training Random Forest classifier...")

    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    report = classification_report(
        y_test,
        y_pred,
        target_names=["authentic", "tampered"],
        output_dict=True,
    )

    text_report = classification_report(
        y_test,
        y_pred,
        target_names=["authentic", "tampered"],
    )

    matrix = confusion_matrix(y_test, y_pred)

    print()
    print("Accuracy:", accuracy)

    print()
    print("Classification Report:")
    print(text_report)

    print()
    print("Confusion Matrix:")
    print(matrix)

    CLASSIFIER_PATH.parent.mkdir(parents=True, exist_ok=True)

    model_performance = {
        "accuracy": float(accuracy),
        "authentic_precision": float(report["authentic"]["precision"]),
        "authentic_recall": float(report["authentic"]["recall"]),
        "authentic_f1": float(report["authentic"]["f1-score"]),
        "tampered_precision": float(report["tampered"]["precision"]),
        "tampered_recall": float(report["tampered"]["recall"]),
        "tampered_f1": float(report["tampered"]["f1-score"]),
        "n_features": len(FEATURE_NAMES),
        "n_trees": 200,
        "train_samples": int(len(y_train)),
        "test_samples": int(len(y_test)),
    }

    model_bundle = {
        "classifier": classifier,
        "accuracy": accuracy,
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
        "model_performance": model_performance,
        "feature_names": FEATURE_NAMES,
        "n_features": len(FEATURE_NAMES),
        "label_names": {
            0: "authentic",
            1: "tampered",
        },
    }

    with open(CLASSIFIER_PATH, "wb") as file:
        pickle.dump(model_bundle, file)

    save_model_performance(model_performance)

    print()
    print("Model saved successfully at:")
    print(CLASSIFIER_PATH)

    print()
    print("Model performance saved in database.")


if __name__ == "__main__":
    train()