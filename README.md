# Document Intelligence System

A document image tampering detection system built using **OpenCV preprocessing**, **Error Level Analysis (ELA)**, **OCR**, **handcrafted forensic features**, **Random Forest classification**, **FastAPI**, **SQLite**, and **Streamlit**.

This project verifies whether an uploaded document image appears **authentic** or **tampered**.

## GitHub Repository

Project link: https://github.com/IshuDevQ/Document_Intelligence

---

## What This Project Does

The system analyzes document images and predicts whether they are likely to be authentic or tampered. It combines image forensics, OCR, handcrafted feature extraction, and machine learning.

For each uploaded image, the system performs:

```text
1. Image preprocessing
2. Error Level Analysis
3. OCR text extraction
4. Forensic feature extraction
5. Random Forest classification
6. Result storage in SQLite
7. Result display through FastAPI and Streamlit
```

---

## Main Features

- Detects possible image tampering in document images
- Uses CASIA 2.0 image tampering dataset
- Applies OpenCV-based preprocessing
- Generates Error Level Analysis maps
- Extracts 18 handcrafted forensic features
- Uses a Random Forest classifier with 200 trees
- Extracts OCR text using Tesseract
- Stores verification history in SQLite
- Displays prediction confidence, probabilities, OCR output, and ELA map
- Tracks model performance metrics

---

## Tech Stack

| Component | Technology |
|---|---|
| Backend API | FastAPI |
| Frontend | Streamlit |
| Database | SQLite |
| Image Processing | OpenCV, Pillow |
| Tampering Analysis | Error Level Analysis |
| OCR | Tesseract OCR, pytesseract |
| Machine Learning | scikit-learn Random Forest |
| Dataset | CASIA 2.0 Image Tampering Detection Dataset |

---

## Dataset

This project uses the **CASIA 2.0 Image Tampering Detection Dataset**.

The dataset contains two main image categories:

```text
Au → authentic images
Tp → tampered images
```

The images are organized into:

```text
data/training_set/authentic/
data/training_set/tampered/
```

---

## Verification Pipeline

```text
Uploaded Image
      ↓
OpenCV Preprocessing
      ↓
Error Level Analysis
      ↓
OCR Extraction
      ↓
18 Feature Extraction
      ↓
Random Forest Prediction
      ↓
FastAPI Response
      ↓
SQLite Storage
      ↓
Streamlit Display
```

---

## Extracted Features

The model uses 18 handcrafted forensic features:

```text
1. ela_mean
2. ela_std
3. ela_max
4. ela_p90
5. ela_p95
6. ela_bright_pct
7. ela_high_pct
8. ela_region_std
9. noise_var
10. edge_density
11. texture_contrast
12. texture_energy
13. color_std_r
14. color_std_g
15. aspect_ratio
16. brightness_mean
17. brightness_std
18. ocr_confidence
```

---

## Model

The classifier is a Random Forest model.

```text
n_estimators = 200
class_weight = balanced
random_state = 42
```

The trained model is saved as:

```text
app/models/classifier.pkl
```

---

## Output

For every uploaded document, the system returns:

```json
{
  "request_id": "abc12345",
  "filename": "document.jpg",
  "label": "tampered",
  "confidence": 0.91,
  "authentic_prob": 0.09,
  "tampered_prob": 0.91,
  "is_uncertain": false,
  "ela_mean": 12.45,
  "ela_region_std": 5.73,
  "ela_image_path": "data/results/ela_abc12345.png",
  "ocr_text": "Extracted document text...",
  "ocr_confidence": 0.82,
  "processing_time_ms": 430,
  "verdict": "Tampered document detected with 91.00% confidence."
}
```

---

## Results

### Training Result on CASIA 2.0

The Random Forest model was trained on the CASIA 2.0 dataset after extracting 18 handcrafted forensic features from each image.

```text
Feature matrix shape: (12614, 18)
Label vector shape: (12614,)
Number of features: 18
```

The final model achieved the following performance on the test split:

```text
Accuracy: 0.6722156163297661
```

Classification report:

```text
              precision    recall  f1-score   support

   authentic       0.71      0.76      0.73      1498
    tampered       0.61      0.54      0.57      1025

    accuracy                           0.67      2523
   macro avg       0.66      0.65      0.65      2523
weighted avg       0.67      0.67      0.67      2523
```

Confusion matrix:

```text
[[1143  355]
 [ 472  553]]
```

Interpretation:

```text
- 1143 authentic images were correctly classified as authentic.
- 355 authentic images were incorrectly classified as tampered.
- 472 tampered images were incorrectly classified as authentic.
- 553 tampered images were correctly classified as tampered.
```

---

## Result Types

### 1. Document Verification Result

Each uploaded document is classified as either:

```text
authentic
tampered
```

The prediction includes:

```text
- predicted label
- confidence score
- authentic probability
- tampered probability
- uncertainty flag
- OCR text
- OCR confidence
- ELA feature values
- generated ELA map path
- processing time
```

### 2. ELA Map Result

For every uploaded image, the system generates an Error Level Analysis map.

Example:

```text
data/results/ela_abc12345.png
```

The ELA map helps visualize regions that may have different compression behavior, which can indicate possible image manipulation.

### 3. Model Performance Result

The system stores model performance metrics such as:

```text
- accuracy
- authentic precision
- authentic recall
- authentic F1-score
- tampered precision
- tampered recall
- tampered F1-score
- number of features
- number of trees
- training sample count
- testing sample count
```

### 4. Verification History Result

Every verification request is saved with:

```text
- request ID
- filename
- predicted label
- confidence
- authentic probability
- tampered probability
- uncertainty status
- ELA statistics
- OCR text
- OCR confidence
- processing time
- timestamp
```

---

## Conclusion

This project demonstrates a complete document image tampering detection pipeline using traditional image forensic analysis and machine learning. It combines ELA, OCR, handcrafted features, Random Forest classification, API development, database logging, and a user-facing Streamlit interface.
