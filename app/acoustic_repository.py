"""Persistence for derived acoustic features, benchmarks, and decisions."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Iterable, Sequence

import numpy as np

from app.database import db_connection


@dataclass(frozen=True)
class StoredRecording:
    id: int
    word: str
    speaker_hash: str
    dataset_split: str
    source_hash: str
    duration_ms: int
    feature: list[float] | None
    quality_status: str
    cleaning_version: str
    extractor_version: str


@dataclass(frozen=True)
class AcousticBenchmark:
    id: int
    word: str
    language: str
    locale: str
    task: str
    prototypes: list[list[float]]
    feature_center: list[float]
    feature_scale: list[float]
    distance_statistics: dict[str, float | int | None]
    accept_threshold: float
    reject_threshold: float
    sample_count: int
    speaker_count: int
    cleaning_version: str
    extractor_version: str
    decision_version: str
    version: str


def store_recording(
    *,
    word: str,
    language: str,
    locale: str,
    speaker_hash: str,
    take_index: int,
    source_hash: str,
    source_member: str,
    dataset_split: str,
    duration_ms: int,
    quality_status: str,
    quality_reasons: Sequence[str],
    feature: Sequence[float] | None,
    cleaning_version: str,
    extractor_version: str,
) -> int:
    """Insert one derived recording, or return its existing id by source hash."""
    feature_json = (
        json.dumps([float(value) for value in feature])
        if feature is not None
        else None
    )
    with db_connection() as database:
        database.execute("BEGIN IMMEDIATE")
        try:
            database.execute(
                """
                INSERT OR IGNORE INTO processed_recordings (
                    word, language, locale, speaker_hash, take_index,
                    source_hash, source_member, dataset_split, duration_ms,
                    quality_status, quality_reasons_json, feature_json,
                    cleaning_version, extractor_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    word.casefold(),
                    language.casefold(),
                    locale,
                    speaker_hash,
                    take_index,
                    source_hash,
                    source_member,
                    dataset_split,
                    duration_ms,
                    quality_status,
                    json.dumps(list(quality_reasons)),
                    feature_json,
                    cleaning_version,
                    extractor_version,
                ),
            )
            row = database.execute(
                "SELECT id FROM processed_recordings WHERE source_hash = ?",
                (source_hash,),
            ).fetchone()
            database.execute("COMMIT")
        except Exception:
            database.execute("ROLLBACK")
            raise
    if row is None:
        raise RuntimeError("Recording upsert returned no row")
    return int(row["id"])


def list_recordings(
    *,
    word: str | None = None,
    locale: str = "en-US",
    dataset_split: str | None = None,
    accepted_only: bool = True,
) -> list[StoredRecording]:
    conditions = ["locale = ?"]
    parameters: list[object] = [locale]
    if word is not None:
        conditions.append("word = ?")
        parameters.append(word.casefold())
    if dataset_split is not None:
        conditions.append("dataset_split = ?")
        parameters.append(dataset_split)
    if accepted_only:
        conditions.append("quality_status = 'ACCEPT'")
        conditions.append("feature_json IS NOT NULL")

    query = f"""
        SELECT id, word, speaker_hash, dataset_split, source_hash, duration_ms,
               feature_json, quality_status, cleaning_version, extractor_version
        FROM processed_recordings
        WHERE {" AND ".join(conditions)}
        ORDER BY word, speaker_hash, source_hash
    """
    with db_connection() as database:
        rows = database.execute(query, parameters).fetchall()
    return [
        StoredRecording(
            id=int(row["id"]),
            word=str(row["word"]),
            speaker_hash=str(row["speaker_hash"]),
            dataset_split=str(row["dataset_split"]),
            source_hash=str(row["source_hash"]),
            duration_ms=int(row["duration_ms"]),
            feature=(
                [float(value) for value in json.loads(row["feature_json"])]
                if row["feature_json"] is not None
                else None
            ),
            quality_status=str(row["quality_status"]),
            cleaning_version=str(row["cleaning_version"]),
            extractor_version=str(row["extractor_version"]),
        )
        for row in rows
    ]


def store_acoustic_benchmark(
    *,
    word: str,
    language: str,
    locale: str,
    task: str,
    prototypes: Iterable[Sequence[float]],
    feature_center: Sequence[float],
    feature_scale: Sequence[float],
    distance_statistics: dict[str, float | int | None],
    accept_threshold: float,
    reject_threshold: float,
    sample_count: int,
    speaker_count: int,
    cleaning_version: str,
    extractor_version: str,
    decision_version: str,
    version: str,
) -> int:
    """Store one immutable benchmark version and activate it."""
    if reject_threshold < accept_threshold:
        raise ValueError("reject_threshold must be >= accept_threshold")
    encoded_prototypes = [
        [float(value) for value in prototype] for prototype in prototypes
    ]
    with db_connection() as database:
        database.execute("BEGIN IMMEDIATE")
        try:
            database.execute(
                """
                UPDATE acoustic_benchmarks
                SET active = 0
                WHERE word = ? AND locale = ? AND task = ?
                """,
                (word.casefold(), locale, task),
            )
            database.execute(
                """
                INSERT INTO acoustic_benchmarks (
                    word, language, locale, task, prototypes_json,
                    feature_center_json, feature_scale_json,
                    distance_statistics_json, accept_threshold,
                    reject_threshold, sample_count, speaker_count,
                    cleaning_version, extractor_version, decision_version,
                    version, active
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
                ON CONFLICT(word, locale, task, version) DO UPDATE SET
                    prototypes_json = excluded.prototypes_json,
                    feature_center_json = excluded.feature_center_json,
                    feature_scale_json = excluded.feature_scale_json,
                    distance_statistics_json = excluded.distance_statistics_json,
                    accept_threshold = excluded.accept_threshold,
                    reject_threshold = excluded.reject_threshold,
                    sample_count = excluded.sample_count,
                    speaker_count = excluded.speaker_count,
                    cleaning_version = excluded.cleaning_version,
                    extractor_version = excluded.extractor_version,
                    decision_version = excluded.decision_version,
                    active = 1
                """,
                (
                    word.casefold(),
                    language.casefold(),
                    locale,
                    task,
                    json.dumps(encoded_prototypes),
                    json.dumps([float(value) for value in feature_center]),
                    json.dumps([float(value) for value in feature_scale]),
                    json.dumps(distance_statistics, sort_keys=True),
                    float(accept_threshold),
                    float(reject_threshold),
                    int(sample_count),
                    int(speaker_count),
                    cleaning_version,
                    extractor_version,
                    decision_version,
                    version,
                ),
            )
            row = database.execute(
                """
                SELECT id FROM acoustic_benchmarks
                WHERE word = ? AND locale = ? AND task = ? AND version = ?
                """,
                (word.casefold(), locale, task, version),
            ).fetchone()
            database.execute("COMMIT")
        except Exception:
            database.execute("ROLLBACK")
            raise
    if row is None:
        raise RuntimeError("Benchmark upsert returned no row")
    return int(row["id"])


def find_acoustic_benchmark(
    word: str,
    *,
    locale: str = "en-US",
    task: str = "word_pronunciation",
) -> AcousticBenchmark | None:
    with db_connection() as database:
        row = database.execute(
            """
            SELECT * FROM acoustic_benchmarks
            WHERE word = ? AND locale = ? AND task = ? AND active = 1
            ORDER BY id DESC
            LIMIT 1
            """,
            (word.casefold(), locale, task),
        ).fetchone()
    if row is None:
        return None
    return AcousticBenchmark(
        id=int(row["id"]),
        word=str(row["word"]),
        language=str(row["language"]),
        locale=str(row["locale"]),
        task=str(row["task"]),
        prototypes=json.loads(row["prototypes_json"]),
        feature_center=json.loads(row["feature_center_json"]),
        feature_scale=json.loads(row["feature_scale_json"]),
        distance_statistics=json.loads(row["distance_statistics_json"]),
        accept_threshold=float(row["accept_threshold"]),
        reject_threshold=float(row["reject_threshold"]),
        sample_count=int(row["sample_count"]),
        speaker_count=int(row["speaker_count"]),
        cleaning_version=str(row["cleaning_version"]),
        extractor_version=str(row["extractor_version"]),
        decision_version=str(row["decision_version"]),
        version=str(row["version"]),
    )


def save_acoustic_attempt(
    *,
    session_id: str,
    benchmark_id: int,
    decision_status: str,
    distance: float | None,
    score: float | None,
    confidence: float | None,
    reason_codes: Sequence[str],
    decision_version: str,
) -> int:
    with db_connection() as database:
        row = database.execute(
            """
            INSERT INTO acoustic_attempts (
                session_id, benchmark_id, decision_status, distance, score,
                confidence, reason_codes_json, decision_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            RETURNING id
            """,
            (
                session_id,
                benchmark_id,
                decision_status,
                distance,
                score,
                confidence,
                json.dumps(list(reason_codes)),
                decision_version,
            ),
        ).fetchone()
    if row is None:
        raise RuntimeError("Acoustic attempt insertion returned no row")
    return int(row["id"])


def feature_matrix(recordings: Sequence[StoredRecording]) -> np.ndarray:
    """Convert stored finite features to a validated two-dimensional matrix."""
    values = [recording.feature for recording in recordings]
    if not values or any(feature is None for feature in values):
        raise ValueError("At least one recording has no feature")
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or not np.all(np.isfinite(matrix)):
        raise ValueError("Stored features must be a finite matrix")
    return matrix
