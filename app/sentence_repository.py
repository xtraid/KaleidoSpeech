"""SQLite persistence for annotated known-sentence benchmarks."""

from __future__ import annotations

import io
import json
from typing import Any

import numpy as np

from app.database import db_connection
from app.sentence_streaming import SentenceBenchmark, SentenceUnit
from app.temporal_features import FEATURE_DIMENSION


def _encode(values: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, np.asarray(values, dtype=np.float32), allow_pickle=False)
    return buffer.getvalue()


def _decode(blob: bytes) -> np.ndarray:
    values = np.load(io.BytesIO(blob), allow_pickle=False)
    if values.ndim != 2 or values.shape[1] != FEATURE_DIMENSION:
        raise ValueError("invalid stored sentence features")
    return values.astype(np.float32, copy=False)


def store_sentence_benchmark(
    sentence_id: str,
    text: str,
    language: str,
    locale: str,
    reference: np.ndarray,
    units: tuple[SentenceUnit, ...],
    *,
    error_threshold: float,
    final_threshold: float,
    version: str = "1",
) -> int:
    payload = [
        {"text": unit.text, "start_ms": unit.start_ms, "end_ms": unit.end_ms}
        for unit in units
    ]
    with db_connection() as database:
        database.execute("BEGIN IMMEDIATE")
        database.execute(
            "UPDATE sentence_benchmarks SET active = 0 WHERE sentence_id = ?",
            (sentence_id,),
        )
        database.execute(
            """
            INSERT INTO sentence_benchmarks (
                sentence_id, text, language, locale, units_json,
                reference_blob, feature_dimension, error_threshold,
                final_threshold, version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(sentence_id, version) DO UPDATE SET
                text = excluded.text,
                language = excluded.language,
                locale = excluded.locale,
                units_json = excluded.units_json,
                reference_blob = excluded.reference_blob,
                feature_dimension = excluded.feature_dimension,
                error_threshold = excluded.error_threshold,
                final_threshold = excluded.final_threshold,
                active = 1
            """,
            (
                sentence_id, text, language, locale,
                json.dumps(payload, ensure_ascii=False), _encode(reference),
                FEATURE_DIMENSION, error_threshold, final_threshold, version,
            ),
        )
        row = database.execute(
            """
            SELECT id FROM sentence_benchmarks
            WHERE sentence_id = ? AND version = ?
            """,
            (sentence_id, version),
        ).fetchone()
        database.execute("COMMIT")
        assert row is not None
        return int(row["id"])


def find_sentence_benchmark(sentence_id: str) -> SentenceBenchmark | None:
    with db_connection() as database:
        row = database.execute(
            """
            SELECT * FROM sentence_benchmarks
            WHERE sentence_id = ? AND active = 1
            ORDER BY id DESC LIMIT 1
            """,
            (sentence_id,),
        ).fetchone()
    if row is None:
        return None
    units = tuple(SentenceUnit(**item) for item in json.loads(row["units_json"]))
    return SentenceBenchmark(
        id=int(row["id"]),
        sentence_id=str(row["sentence_id"]),
        text=str(row["text"]),
        language=str(row["language"]),
        locale=str(row["locale"]),
        reference=_decode(row["reference_blob"]),
        units=units,
        error_threshold=float(row["error_threshold"]),
        final_threshold=float(row["final_threshold"]),
        version=str(row["version"]),
    )


def save_sentence_attempt(
    session_id: str,
    benchmark_id: int,
    result: dict[str, Any],
    errors: list[dict[str, Any]],
) -> int:
    with db_connection() as database:
        cursor = database.execute(
            """
            INSERT INTO sentence_attempts (
                session_id, benchmark_id, status, distance,
                errors_json, model_version
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                benchmark_id,
                result["status"],
                result["distance"],
                json.dumps(errors, ensure_ascii=False),
                result["model_version"],
            ),
        )
        return int(cursor.lastrowid)
