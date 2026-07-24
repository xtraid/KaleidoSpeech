"""SQLite repository for benchmarks and final pronunciation attempts."""

import json
from dataclasses import dataclass
from typing import Protocol

from app.database import db_connection


@dataclass(frozen=True)
class Benchmark:
    id: int
    language: str
    word: str
    expected_phonemes: list[str]
    accepted_variants: list[list[str]]
    version: str


@dataclass(frozen=True)
class Attempt:
    id: int
    session_id: str
    benchmark_id: int
    detected_phonemes: list[str]
    confidence: float
    score: float | None
    engine_version: str
    created_at: str


class AttemptResult(Protocol):
    detected_phonemes: list[str]
    overall_confidence: float
    score: float | None
    engine_version: str


def find_benchmark(language: str, word: str) -> Benchmark | None:
    with db_connection() as database:
        row = database.execute(
            """
            SELECT id, language, word, expected_phonemes_json,
                   accepted_variants_json, version
            FROM benchmarks
            WHERE language = ? AND word = ? AND active = 1
            ORDER BY version DESC
            LIMIT 1
            """,
            (language.casefold(), word.casefold()),
        ).fetchone()

    if row is None:
        return None

    return Benchmark(
        id=row["id"],
        language=row["language"],
        word=row["word"],
        expected_phonemes=json.loads(row["expected_phonemes_json"]),
        accepted_variants=json.loads(row["accepted_variants_json"]),
        version=row["version"],
    )


def save_attempt(
    session_id: str,
    benchmark_id: int,
    result: AttemptResult,
) -> Attempt:
    detected_phonemes_json = json.dumps(
        result.detected_phonemes,
        ensure_ascii=False,
    )

    with db_connection() as database:
        database.execute("BEGIN IMMEDIATE")
        try:
            row = database.execute(
                """
                INSERT INTO attempts (
                    session_id,
                    benchmark_id,
                    detected_phonemes_json,
                    confidence,
                    score,
                    engine_version
                ) VALUES (?, ?, ?, ?, ?, ?)
                RETURNING id, created_at
                """,
                (
                    session_id,
                    benchmark_id,
                    detected_phonemes_json,
                    result.overall_confidence,
                    result.score,
                    result.engine_version,
                ),
            ).fetchone()
            database.execute("COMMIT")
        except Exception:
            database.execute("ROLLBACK")
            raise

    if row is None:
        raise RuntimeError("Attempt insertion returned no row")

    return Attempt(
        id=row["id"],
        session_id=session_id,
        benchmark_id=benchmark_id,
        detected_phonemes=result.detected_phonemes,
        confidence=result.overall_confidence,
        score=result.score,
        engine_version=result.engine_version,
        created_at=row["created_at"],
    )
