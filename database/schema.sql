CREATE TABLE IF NOT EXISTS verifications (
    id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    label TEXT NOT NULL,
    confidence REAL NOT NULL,
    authentic_prob REAL,
    tampered_prob REAL,
    is_uncertain INTEGER DEFAULT 0,
    ela_mean REAL,
    ela_region_std REAL,
    ela_image_path TEXT,
    ocr_text TEXT,
    ocr_confidence REAL,
    processing_time_ms INTEGER,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS model_performance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    accuracy REAL,
    authentic_precision REAL,
    authentic_recall REAL,
    authentic_f1 REAL,
    tampered_precision REAL,
    tampered_recall REAL,
    tampered_f1 REAL,
    n_features INTEGER,
    n_trees INTEGER,
    train_samples INTEGER,
    test_samples INTEGER,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);