from pydantic import BaseModel


class VerificationResponse(BaseModel):
    request_id: str
    filename: str
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
    verdict: str


class HistoryResponse(BaseModel):
    total: int
    results: list[dict]


class StatsResponse(BaseModel):
    total: int = 0
    total_authentic: int = 0
    total_tampered: int = 0
    total_uncertain: int = 0
    avg_confidence: float = 0.0
    avg_processing_ms: float = 0.0


class ModelPerformanceResponse(BaseModel):
    id: int | None = None
    accuracy: float = 0.0
    authentic_precision: float = 0.0
    authentic_recall: float = 0.0
    authentic_f1: float = 0.0
    tampered_precision: float = 0.0
    tampered_recall: float = 0.0
    tampered_f1: float = 0.0
    n_features: int = 0
    n_trees: int = 0
    train_samples: int = 0
    test_samples: int = 0
    created_at: str | None = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    database_ready: bool