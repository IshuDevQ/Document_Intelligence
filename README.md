
# Document Intelligence System

A document image tampering detection system built using **OpenCV preprocessing**, **Error Level Analysis (ELA)**, **OCR**, **handcrafted forensic features**, **Random Forest classification**, **FastAPI**, **SQLite**, and **Streamlit**.

This project verifies whether an uploaded document image appears **authentic** or **tampered**.

## GitHub Repository

Project link: https://github.com/IshuDevQ/Document_Intelligence

---

## Features

- CASIA 2.0 dataset download using KaggleHub
- Automatic dataset organization into authentic and tampered folders
- OpenCV-based preprocessing:
  - resizing
  - deskewing
  - denoising
  - brightness and contrast normalization
- Error Level Analysis image generation
- 18 handcrafted forensic features
- Random Forest classifier with 200 trees
- OCR using Tesseract
- FastAPI backend
- Streamlit frontend
- SQLite verification history
- Model performance tracking
- ELA map visualization in the frontend

---

## Project Structure

```text
document-intelligence-system/
│
├── requirements.txt
├── download_dataset.py
│
├── app/
│   ├── __init__.py
│   ├── config.py
│   │
│   ├── models/
│   │   └── .gitkeep
│   │
│   └── training/
│       ├── __init__.py
│       └── train.py
│
├── pipeline/
│   ├── __init__.py
│   ├── preprocessor.py
│   ├── ela_analyzer.py
│   ├── feature_extractor.py
│   └── verifier.py
│
├── database/
│   ├── __init__.py
│   ├── schema.sql
│   └── db.py
│
├── api/
│   ├── __init__.py
│   ├── schemas.py
│   └── main.py
│
├── streamlit_app/
│   └── app.py
│
└── data/
    ├── results/
    │   └── .gitkeep
    │
    └── training_set/
        ├── authentic/
        │   └── .gitkeep
        └── tampered/
            └── .gitkeep
```

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

This project uses the CASIA 2.0 Image Tampering Detection Dataset.

The dataset is downloaded using KaggleHub:

```python
import kagglehub

path = kagglehub.dataset_download(
    "divg07/casia-20-image-tampering-detection-dataset"
)

print("Path to dataset files:", path)
```

The downloader script automatically arranges the dataset into:

```text
data/training_set/authentic/
data/training_set/tampered/
```

where:

```text
Au → authentic images
Tp → tampered images
```

The dataset itself is not pushed to GitHub because it is large.

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/IshuDevQ/Document_Intelligence.git
cd Document_Intelligence
```

---

### 2. Create Virtual Environment

```bash
python -m venv venv
```

Activate it.

#### Windows

```bash
venv\\Scripts\\activate
```

#### macOS / Linux

```bash
source venv/bin/activate
```

---

### 3. Install Python Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Install Tesseract OCR

This project uses Tesseract for OCR extraction.

### Windows

Install Tesseract from:

```text
https://github.com/UB-Mannheim/tesseract/wiki
```

Usually, Tesseract is installed at:

```text
C:\\Program Files\\Tesseract-OCR\\tesseract.exe
```

Add this folder to your system PATH:

```text
C:\\Program Files\\Tesseract-OCR
```

Then verify:

```bash
tesseract --version
```

### Ubuntu / Debian

```bash
sudo apt update
sudo apt install tesseract-ocr
```

### macOS

```bash
brew install tesseract
```

---

## Download and Prepare Dataset

Run:

```bash
python download_dataset.py
```

After this, your dataset should be arranged like this:

```text
data/
└── training_set/
    ├── authentic/
    │   ├── Au_*.jpg
    │   └── ...
    │
    └── tampered/
        ├── Tp_*.jpg
        └── ...
```

---

## Train the Model

Run:

```bash
python -m app.training.train
```

This will:

1. Load images from `data/training_set/authentic/`
2. Load images from `data/training_set/tampered/`
3. Preprocess each image
4. Generate ELA maps
5. Extract 18 forensic features
6. Train a Random Forest classifier
7. Save the model to:

```text
app/models/classifier.pkl
```

It also stores model performance metrics in SQLite.

---

## Run the FastAPI Backend

Start the backend server:

```bash
uvicorn api.main:app --reload --port 8000
```

Open API documentation:

```text
http://localhost:8000/docs
```

---

## Run the Streamlit Frontend

Open another terminal, activate the same virtual environment, then run:

```bash
streamlit run streamlit_app/app.py
```

Open the app:

```text
http://localhost:8501
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | API home |
| GET | `/health` | Check backend and model status |
| POST | `/verify` | Upload document image and verify tampering |
| GET | `/history` | View verification history |
| GET | `/stats` | View verification statistics |
| GET | `/model-performance` | View latest model performance metrics |

---

## Verification Pipeline

The project follows this pipeline:

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

The model uses 18 handcrafted features:

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

The classifier is a Random Forest model:

```text
n_estimators = 200
class_weight = balanced
random_state = 42
```

The trained model is saved as:

```text
app/models/classifier.pkl
```

The model file is ignored by Git because it can be regenerated by training.

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

## Files Not Pushed to GitHub

The following files and folders should not be committed:

```text
venv/
data/training_set/authentic/*
data/training_set/tampered/*
data/results/*
app/models/classifier.pkl
database/doc_intel.db
__pycache__/
.env
kaggle.json
```

Use `.gitkeep` files to preserve empty folders.

---

## Recommended `.gitignore`

```gitignore
# Python
__pycache__/
*.pyc
*.pyo
*.pyd

# Virtual environments
venv/
env/
.venv/

# Dataset
data/training_set/authentic/*
data/training_set/tampered/*
!data/training_set/authentic/.gitkeep
!data/training_set/tampered/.gitkeep

# Generated ELA results
data/results/*
!data/results/.gitkeep

# Trained models
app/models/*
!app/models/.gitkeep

# SQLite database
database/*.db
database/*.sqlite
database/*.sqlite3

# Kaggle credentials
kaggle.json

# Environment variables
.env

# OS/editor files
.DS_Store
.vscode/
.idea/

# Streamlit secrets
.streamlit/secrets.toml
```

---

## How to Run Full Project

Use this sequence:

```bash
python -m venv venv
```

Windows:

```bash
venv\\Scripts\\activate
```

macOS / Linux:

```bash
source venv/bin/activate
```

Then:

```bash
pip install -r requirements.txt
python download_dataset.py
python -m app.training.train
uvicorn api.main:app --reload --port 8000
```

In another terminal:

```bash
streamlit run streamlit_app/app.py
```

---

## Notes

- The dataset is not included in the repository.
- The trained model is not included in the repository.
- The SQLite database is generated automatically.
- ELA result images are generated automatically during verification.
- You must train the model before using the `/verify` endpoint.

---

## License

This project is for educational and research purposes.