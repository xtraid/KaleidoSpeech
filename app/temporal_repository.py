"""SQLite persistence for temporal features and DTW benchmark artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import json
from typing import Sequence
import zlib

import numpy as np

from app.database import db_connection
from app.temporal_features import FEATURE_DIMENSION


@dataclass(frozen=True)
class TemporalRecording:
    recording_id: int
    word: str
    speaker_hash: str
    dataset_split: str
    sequence: np.ndarray


@dataclass(frozen=True)
class TemporalBenchmark:
    id: int
    word: str
    language: str
    locale: str
    task: str
    prototype_recording_ids: tuple[int, ...]
    prototypes: tuple[np.ndarray, ...]
    accept_threshold: float
    reject_threshold: float
    margin_threshold: float
    distance_statistics: dict[str, float | int | None]
    sample_count: int
    speaker_count: int
    extractor_version: str
    distance_version: str
    decision_version: str
    version: str


def _encode_sequence(sequence: np.ndarray) -> bytes:
    values = np.asarray(sequence, dtype="<f4")
    if (
        values.ndim != 2
        or values.shape[1] != FEATURE_DIMENSION
        or not np.all(np.isfinite(values))
    ):
        raise ValueError("Invalid temporal feature sequence")
    return zlib.compress(values.tobytes(), level=6)


def _decode_sequence(blob: bytes, frames: int, dimension: int) -> np.ndarray:
    if frames <= 0 or dimension != FEATURE_DIMENSION:
        raise ValueError("Invalid temporal feature shape metadata")
    raw = zlib.decompress(blob)
    expected = frames * dimension * np.dtype("<f4").itemsize
    if len(raw) != expected:
        raise ValueError("Temporal feature blob does not match its shape")
    values = np.frombuffer(raw, dtype="<f4").reshape(frames, dimension).copy()
    if not np.all(np.isfinite(values)):
        raise ValueError("Temporal feature blob contains non-finite values")
    return values


def _encode_prototypes(prototypes: Sequence[np.ndarray]) -> bytes:
    if not prototypes:
        raise ValueError("At least one DTW prototype is required")
    output = BytesIO()
    np.savez_compressed(
        output,
        **{
            f"prototype_{index}": np.asarray(prototype, dtype="<f4")
            for index, prototype in enumerate(prototypes)
        },
    )
    return output.getvalue()


def _decode_prototypes(blob: bytes) -> tuple[np.ndarray, ...]:
    with np.load(BytesIO(blob), allow_pickle=False) as archive:
        keys = sorted(
            archive.files,
            key=lambda value: int(value.rsplit("_", 1)[1]),
        )
        prototypes = tuple(
            np.asarray(archive[key], dtype=np.float32) for key in keys
        )
    for prototype in prototypes:
        if (
            prototype.ndim != 2
            or prototype.shape[1] != FEATURE_DIMENSION
            or not np.all(np.isfinite(prototype))
        ):
            raise ValueError("Invalid prototype artifact")
    if not prototypes:
        raise ValueError("Prototype artifact is empty")
    return prototypes


def store_temporal_feature(
    recording_id: int,
    sequence: np.ndarray,
    *,
    extractor_version: str,
    distance_version: str,
) -> None:
    values = np.asarray(sequence, dtype=np.float32)
    encoded = _encode_sequence(values)
    with db_connection() as database:
        database.execute(
            """
            INSERT INTO temporal_features (
                recording_id, feature_blob, frame_count, feature_dimension,
                extractor_version, distance_version
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(recording_id) DO UPDATE SET
                feature_blob = excluded.feature_blob,
                frame_count = excluded.frame_count,
                feature_dimension = excluded.feature_dimension,
                extractor_version = excluded.extractor_version,
                distance_version = excluded.distance_version
            """,
            (
                recording_id,
                encoded,
                values.shape[0],
                values.shape[1],
                extractor_version,
                distance_version,
            ),
        )


def list_temporal_recordings(
    *,
    word: str | None = None,
    locale: str = "en-US",
    dataset_split: str | None = None,
) -> list[TemporalRecording]:
    conditions = [
        "recording.locale = ?",
        "recording.quality_status = 'ACCEPT'",
    ]
    parameters: list[object] = [locale]
    if word is not None:
        conditions.append("recording.word = ?")
        parameters.append(word.casefold())
    if dataset_split is not None:
        conditions.append("recording.dataset_split = ?")
        parameters.append(dataset_split)
    with db_connection() as database:
        rows = database.execute(
            f"""
            SELECT feature.recording_id, feature.feature_blob,
                   feature.frame_count, feature.feature_dimension,
                   recording.word, recording.speaker_hash,
                   recording.dataset_split
            FROM temporal_features AS feature
            JOIN processed_recordings AS recording
              ON recording.id = feature.recording_id
            WHERE {" AND ".join(conditions)}
            ORDER BY recording.word, recording.speaker_hash,
                     feature.recording_id
            """,
            parameters,
        ).fetchall()
    return [
        TemporalRecording(
            recording_id=int(row["recording_id"]),
            word=str(row["word"]),
            speaker_hash=str(row["speaker_hash"]),
            dataset_split=str(row["dataset_split"]),
            sequence=_decode_sequence(
                row["feature_blob"],
                int(row["frame_count"]),
                int(row["feature_dimension"]),
            ),
        )
        for row in rows
    ]


def store_temporal_benchmark(
    *,
    word: str,
    language: str,
    locale: str,
    task: str,
    prototype_recording_ids: Sequence[int],
    prototypes: Sequence[np.ndarray],
    accept_threshold: float,
    reject_threshold: float,
    margin_threshold: float,
    distance_statistics: dict[str, float | int | None],
    sample_count: int,
    speaker_count: int,
    extractor_version: str,
    distance_version: str,
    decision_version: str,
    version: str,
) -> int:
    if reject_threshold <= accept_threshold or margin_threshold < 0:
        raise ValueError("Invalid temporal decision thresholds")
    if len(prototype_recording_ids) != len(prototypes):
        raise ValueError("Prototype IDs and sequences must have equal length")
    artifact = _encode_prototypes(prototypes)
    with db_connection() as database:
        database.execute("BEGIN IMMEDIATE")
        try:
            database.execute(
                """
                UPDATE temporal_benchmarks SET active = 0
                WHERE word = ? AND locale = ? AND task = ?
                """,
                (word.casefold(), locale, task),
            )
            database.execute(
                """
                INSERT INTO temporal_benchmarks (
                    word, language, locale, task,
                    prototype_recording_ids_json, prototypes_blob,
                    accept_threshold, reject_threshold, margin_threshold,
                    distance_statistics_json, sample_count, speaker_count,
                    extractor_version, distance_version, decision_version,
                    version, active
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
                ON CONFLICT(word, locale, task, version) DO UPDATE SET
                    prototype_recording_ids_json =
                        excluded.prototype_recording_ids_json,
                    prototypes_blob = excluded.prototypes_blob,
                    accept_threshold = excluded.accept_threshold,
                    reject_threshold = excluded.reject_threshold,
                    margin_threshold = excluded.margin_threshold,
                    distance_statistics_json =
                        excluded.distance_statistics_json,
                    sample_count = excluded.sample_count,
                    speaker_count = excluded.speaker_count,
                    extractor_version = excluded.extractor_version,
                    distance_version = excluded.distance_version,
                    decision_version = excluded.decision_version,
                    active = 1
                """,
                (
                    word.casefold(),
                    language.casefold(),
                    locale,
                    task,
                    json.dumps([int(value) for value in prototype_recording_ids]),
                    artifact,
                    float(accept_threshold),
                    float(reject_threshold),
                    float(margin_threshold),
                    json.dumps(distance_statistics, sort_keys=True),
                    int(sample_count),
                    int(speaker_count),
                    extractor_version,
                    distance_version,
                    decision_version,
                    version,
                ),
            )
            row = database.execute(
                """
                SELECT id FROM temporal_benchmarks
                WHERE word = ? AND locale = ? AND task = ? AND version = ?
                """,
                (word.casefold(), locale, task, version),
            ).fetchone()
            database.execute("COMMIT")
        except Exception:
            database.execute("ROLLBACK")
            raise
    if row is None:
        raise RuntimeError("Temporal benchmark upsert returned no row")
    return int(row["id"])


def list_temporal_benchmarks(
    *,
    locale: str = "en-US",
    task: str = "word_pronunciation",
) -> list[TemporalBenchmark]:
    with db_connection() as database:
        rows = database.execute(
            """
            SELECT * FROM temporal_benchmarks
            WHERE locale = ? AND task = ? AND active = 1
            ORDER BY word, id DESC
            """,
            (locale, task),
        ).fetchall()
    return [_benchmark_from_row(row) for row in rows]


def find_temporal_benchmark(
    word: str,
    *,
    locale: str = "en-US",
    task: str = "word_pronunciation",
) -> TemporalBenchmark | None:
    with db_connection() as database:
        row = database.execute(
            """
            SELECT * FROM temporal_benchmarks
            WHERE word = ? AND locale = ? AND task = ? AND active = 1
            ORDER BY id DESC LIMIT 1
            """,
            (word.casefold(), locale, task),
        ).fetchone()
    return None if row is None else _benchmark_from_row(row)


def _benchmark_from_row(row) -> TemporalBenchmark:
    return TemporalBenchmark(
        id=int(row["id"]),
        word=str(row["word"]),
        language=str(row["language"]),
        locale=str(row["locale"]),
        task=str(row["task"]),
        prototype_recording_ids=tuple(
            int(value)
            for value in json.loads(row["prototype_recording_ids_json"])
        ),
        prototypes=_decode_prototypes(row["prototypes_blob"]),
        accept_threshold=float(row["accept_threshold"]),
        reject_threshold=float(row["reject_threshold"]),
        margin_threshold=float(row["margin_threshold"]),
        distance_statistics=json.loads(row["distance_statistics_json"]),
        sample_count=int(row["sample_count"]),
        speaker_count=int(row["speaker_count"]),
        extractor_version=str(row["extractor_version"]),
        distance_version=str(row["distance_version"]),
        decision_version=str(row["decision_version"]),
        version=str(row["version"]),
    )


def save_temporal_attempt(
    *,
    session_id: str,
    benchmark_id: int,
    decision_status: str,
    target_distance: float | None,
    competitor_word: str | None,
    competitor_distance: float | None,
    margin: float | None,
    score: float | None,
    reason_codes: Sequence[str],
    decision_version: str,
) -> int:
    with db_connection() as database:
        row = database.execute(
            """
            INSERT INTO temporal_attempts (
                session_id, benchmark_id, decision_status, target_distance,
                competitor_word, competitor_distance, margin, score,
                reason_codes_json, decision_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            RETURNING id
            """,
            (
                session_id,
                benchmark_id,
                decision_status,
                target_distance,
                competitor_word,
                competitor_distance,
                margin,
                score,
                json.dumps(list(reason_codes)),
                decision_version,
            ),
        ).fetchone()
    if row is None:
        raise RuntimeError("Temporal attempt insertion returned no row")
    return int(row["id"])
