from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent

CLASSIFIER_PATH = ROOT_DIR / "app" / "models" / "classifier.pkl"

DB_PATH = ROOT_DIR / "database" / "doc_intel.db"

TRAINING_DATA_DIR = ROOT_DIR / "data" / "training_set"
AUTHENTIC_DIR = TRAINING_DATA_DIR / "authentic"
TAMPERED_DIR = TRAINING_DATA_DIR / "tampered"

RESULTS_DIR = ROOT_DIR / "data" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ELA_QUALITY = 90
ELA_AMPLIFY = 15
IMAGE_RESIZE = (512, 512)

BRIGHT_THRESHOLD = 20
HIGH_THRESHOLD = 50
BLOCK_SIZE = 32

LABEL_AUTHENTIC = 0
LABEL_TAMPERED = 1

LABEL_NAMES = {
    0: "authentic",
    1: "tampered",
}

CONFIDENCE_THRESHOLD = 0.65

VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}

MAX_FILE_SIZE_MB = 10