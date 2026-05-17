import sqlite3
from pathlib import Path

from app.config import DB_PATH


SCHEMA_PATH = Path(__file__).parent / "schema.sql"


def get_connection() -> sqlite3.Connection:
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    schema_sql = SCHEMA_PATH.read_text()

    with get_connection() as connection:
        connection.executescript(schema_sql)


def save_verification(result, filename: str):
    sql = """
        INSERT INTO verifications (
            id,
            filename,
            label,
            confidence,
            authentic_prob,
            tampered_prob,
            is_uncertain,
            ela_mean,
            ela_region_std,
            ela_image_path,
            ocr_text,
            ocr_confidence,
            processing_time_ms
        )
        VALUES (
            :id,
            :filename,
            :label,
            :confidence,
            :authentic_prob,
            :tampered_prob,
            :is_uncertain,
            :ela_mean,
            :ela_region_std,
            :ela_image_path,
            :ocr_text,
            :ocr_confidence,
            :processing_time_ms
        )
    """

    values = {
        "id": result.request_id,
        "filename": filename,
        "label": result.label,
        "confidence": result.confidence,
        "authentic_prob": result.authentic_prob,
        "tampered_prob": result.tampered_prob,
        "is_uncertain": int(result.is_uncertain),
        "ela_mean": result.ela_mean,
        "ela_region_std": result.ela_region_std,
        "ela_image_path": result.ela_image_path,
        "ocr_text": result.ocr_text,
        "ocr_confidence": result.ocr_confidence,
        "processing_time_ms": result.processing_time_ms,
    }

    with get_connection() as connection:
        connection.execute(sql, values)


def get_history(limit: int = 20) -> list[dict]:
    sql = """
        SELECT *
        FROM verifications
        ORDER BY created_at DESC
        LIMIT ?
    """

    with get_connection() as connection:
        rows = connection.execute(sql, (limit,)).fetchall()

    return [dict(row) for row in rows]


def get_stats() -> dict:
    sql = """
        SELECT
            COUNT(*) AS total,
            SUM(CASE WHEN label='authentic' THEN 1 ELSE 0 END) AS total_authentic,
            SUM(CASE WHEN label='tampered' THEN 1 ELSE 0 END) AS total_tampered,
            SUM(CASE WHEN is_uncertain=1 THEN 1 ELSE 0 END) AS total_uncertain,
            ROUND(AVG(confidence), 4) AS avg_confidence,
            ROUND(AVG(processing_time_ms), 1) AS avg_processing_ms
        FROM verifications
    """

    with get_connection() as connection:
        row = connection.execute(sql).fetchone()

    result = dict(row)

    for key, value in result.items():
        if value is None:
            result[key] = 0

    return result


def save_model_performance(metrics: dict):
    init_db()

    sql = """
        INSERT INTO model_performance (
            accuracy,
            authentic_precision,
            authentic_recall,
            authentic_f1,
            tampered_precision,
            tampered_recall,
            tampered_f1,
            n_features,
            n_trees,
            train_samples,
            test_samples
        )
        VALUES (
            :accuracy,
            :authentic_precision,
            :authentic_recall,
            :authentic_f1,
            :tampered_precision,
            :tampered_recall,
            :tampered_f1,
            :n_features,
            :n_trees,
            :train_samples,
            :test_samples
        )
    """

    with get_connection() as connection:
        connection.execute(sql, metrics)


def get_latest_model_performance() -> dict:
    init_db()

    sql = """
        SELECT *
        FROM model_performance
        ORDER BY created_at DESC
        LIMIT 1
    """

    with get_connection() as connection:
        row = connection.execute(sql).fetchone()

    if row is None:
        return {}

    return dict(row)