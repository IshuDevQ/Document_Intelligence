from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import (
    VerificationResponse,
    HistoryResponse,
    StatsResponse,
    HealthResponse,
    ModelPerformanceResponse,
)

from app.config import VALID_EXTENSIONS, MAX_FILE_SIZE_MB
from pipeline.verifier import verify_document, CLASSIFIER
from database.db import (
    init_db,
    save_verification,
    get_history,
    get_stats,
    get_latest_model_performance,
)


app = FastAPI(
    title="Document Intelligence System",
    description="Image tampering detection using ELA and Random Forest",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event():
    init_db()


@app.get("/", response_model=dict)
def home():
    return {
        "message": "Document Intelligence System API is running",
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        model_loaded=CLASSIFIER is not None,
        database_ready=True,
    )


@app.post("/verify", response_model=VerificationResponse)
async def verify(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No file uploaded.",
        )

    suffix = "." + file.filename.split(".")[-1].lower()

    if suffix not in VALID_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {suffix}",
        )

    file_bytes = await file.read()

    size_mb = len(file_bytes) / (1024 * 1024)

    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum allowed size is {MAX_FILE_SIZE_MB} MB.",
        )

    try:
        result = verify_document(file_bytes)

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        )

    save_verification(result, filename=file.filename)

    if result.is_uncertain:
        verdict = (
            f"Uncertain result. Confidence is only {result.confidence:.2%}. "
            f"Manual review recommended."
        )

    elif result.label == "tampered":
        verdict = (
            f"Tampered document detected with {result.confidence:.2%} confidence."
        )

    else:
        verdict = (
            f"Document appears authentic with {result.confidence:.2%} confidence."
        )

    return VerificationResponse(
        request_id=result.request_id,
        filename=file.filename,
        label=result.label,
        confidence=result.confidence,
        authentic_prob=result.authentic_prob,
        tampered_prob=result.tampered_prob,
        is_uncertain=result.is_uncertain,
        ela_mean=result.ela_mean,
        ela_region_std=result.ela_region_std,
        ela_image_path=result.ela_image_path,
        ocr_text=result.ocr_text,
        ocr_confidence=result.ocr_confidence,
        processing_time_ms=result.processing_time_ms,
        verdict=verdict,
    )


@app.get("/history", response_model=HistoryResponse)
def history(limit: int = Query(default=20, ge=1, le=100)):
    results = get_history(limit)

    return HistoryResponse(
        total=len(results),
        results=results,
    )


@app.get("/stats", response_model=StatsResponse)
def stats():
    data = get_stats()

    return StatsResponse(**data)


@app.get("/model-performance", response_model=ModelPerformanceResponse)
def model_performance():
    data = get_latest_model_performance()

    if not data:
        return ModelPerformanceResponse()

    return ModelPerformanceResponse(**data)