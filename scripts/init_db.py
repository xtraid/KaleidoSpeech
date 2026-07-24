"""Create the SQLite schema and seed the initial benchmark."""

import sqlite3

from app.config import get_settings
from app.database import db_connection


SCHEMA = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS benchmarks (
    id INTEGER PRIMARY KEY,
    language TEXT NOT NULL,
    word TEXT NOT NULL,
    expected_phonemes_json TEXT NOT NULL,
    accepted_variants_json TEXT NOT NULL DEFAULT '[]',
    difficulty INTEGER NOT NULL DEFAULT 1,
    version TEXT NOT NULL,
    active INTEGER NOT NULL DEFAULT 1,
    UNIQUE(language, word, version)
);

CREATE TABLE IF NOT EXISTS attempts (
    id INTEGER PRIMARY KEY,
    session_id TEXT NOT NULL,
    benchmark_id INTEGER NOT NULL,
    detected_phonemes_json TEXT,
    confidence REAL,
    score REAL,
    engine_version TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (benchmark_id) REFERENCES benchmarks(id)
);

CREATE INDEX IF NOT EXISTS idx_benchmark_lookup
ON benchmarks(language, word, active);

CREATE INDEX IF NOT EXISTS idx_attempt_session
ON attempts(session_id, created_at);

CREATE TABLE IF NOT EXISTS processed_recordings (
    id INTEGER PRIMARY KEY,
    word TEXT NOT NULL,
    language TEXT NOT NULL,
    locale TEXT NOT NULL,
    speaker_hash TEXT NOT NULL,
    take_index INTEGER NOT NULL,
    source_hash TEXT NOT NULL UNIQUE,
    source_member TEXT NOT NULL,
    dataset_split TEXT NOT NULL
        CHECK(dataset_split IN ('train', 'validation', 'test')),
    duration_ms INTEGER NOT NULL,
    quality_status TEXT NOT NULL,
    quality_reasons_json TEXT NOT NULL DEFAULT '[]',
    feature_json TEXT,
    cleaning_version TEXT NOT NULL,
    extractor_version TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_recording_word_split
ON processed_recordings(word, locale, dataset_split, quality_status);

CREATE INDEX IF NOT EXISTS idx_recording_speaker
ON processed_recordings(speaker_hash, dataset_split);

CREATE TABLE IF NOT EXISTS acoustic_benchmarks (
    id INTEGER PRIMARY KEY,
    word TEXT NOT NULL,
    language TEXT NOT NULL,
    locale TEXT NOT NULL,
    task TEXT NOT NULL,
    prototypes_json TEXT NOT NULL,
    feature_center_json TEXT NOT NULL,
    feature_scale_json TEXT NOT NULL,
    distance_statistics_json TEXT NOT NULL,
    accept_threshold REAL NOT NULL,
    reject_threshold REAL NOT NULL,
    sample_count INTEGER NOT NULL,
    speaker_count INTEGER NOT NULL,
    cleaning_version TEXT NOT NULL,
    extractor_version TEXT NOT NULL,
    decision_version TEXT NOT NULL,
    version TEXT NOT NULL,
    active INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CHECK(reject_threshold >= accept_threshold),
    UNIQUE(word, locale, task, version)
);

CREATE INDEX IF NOT EXISTS idx_acoustic_benchmark_lookup
ON acoustic_benchmarks(word, locale, task, active);

CREATE TABLE IF NOT EXISTS acoustic_attempts (
    id INTEGER PRIMARY KEY,
    session_id TEXT NOT NULL,
    benchmark_id INTEGER NOT NULL,
    decision_status TEXT NOT NULL,
    distance REAL,
    score REAL,
    confidence REAL,
    reason_codes_json TEXT NOT NULL,
    decision_version TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (benchmark_id) REFERENCES acoustic_benchmarks(id)
);

CREATE INDEX IF NOT EXISTS idx_acoustic_attempt_session
ON acoustic_attempts(session_id, created_at);

CREATE TABLE IF NOT EXISTS temporal_features (
    recording_id INTEGER PRIMARY KEY,
    feature_blob BLOB NOT NULL,
    frame_count INTEGER NOT NULL,
    feature_dimension INTEGER NOT NULL,
    extractor_version TEXT NOT NULL,
    distance_version TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (recording_id) REFERENCES processed_recordings(id)
);

CREATE TABLE IF NOT EXISTS temporal_benchmarks (
    id INTEGER PRIMARY KEY,
    word TEXT NOT NULL,
    language TEXT NOT NULL,
    locale TEXT NOT NULL,
    task TEXT NOT NULL,
    prototype_recording_ids_json TEXT NOT NULL,
    prototypes_blob BLOB NOT NULL,
    accept_threshold REAL NOT NULL,
    reject_threshold REAL NOT NULL,
    margin_threshold REAL NOT NULL,
    distance_statistics_json TEXT NOT NULL,
    sample_count INTEGER NOT NULL,
    speaker_count INTEGER NOT NULL,
    extractor_version TEXT NOT NULL,
    distance_version TEXT NOT NULL,
    decision_version TEXT NOT NULL,
    version TEXT NOT NULL,
    active INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CHECK(reject_threshold > accept_threshold),
    CHECK(margin_threshold >= 0),
    UNIQUE(word, locale, task, version)
);

CREATE INDEX IF NOT EXISTS idx_temporal_benchmark_lookup
ON temporal_benchmarks(word, locale, task, active);

CREATE TABLE IF NOT EXISTS temporal_attempts (
    id INTEGER PRIMARY KEY,
    session_id TEXT NOT NULL,
    benchmark_id INTEGER NOT NULL,
    decision_status TEXT NOT NULL,
    target_distance REAL,
    competitor_word TEXT,
    competitor_distance REAL,
    margin REAL,
    score REAL,
    reason_codes_json TEXT NOT NULL,
    decision_version TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (benchmark_id) REFERENCES temporal_benchmarks(id)
);

CREATE TABLE IF NOT EXISTS pediatric_clients (
    id TEXT PRIMARY KEY,
    pseudonym TEXT NOT NULL,
    age_band TEXT NOT NULL
        CHECK(age_band IN ('3-5', '6-8', '9-12', '13-17')),
    primary_language TEXT NOT NULL,
    locale TEXT NOT NULL,
    consent_status TEXT NOT NULL
        CHECK(consent_status IN ('granted', 'withdrawn')),
    consent_recorded_at TEXT NOT NULL,
    retention_until TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS therapy_exercises (
    id TEXT PRIMARY KEY,
    client_id TEXT NOT NULL,
    clinician_id TEXT NOT NULL,
    level TEXT NOT NULL
        CHECK(level IN ('sound', 'syllable', 'word', 'phrase')),
    language TEXT NOT NULL,
    locale TEXT NOT NULL,
    target_text TEXT NOT NULL,
    target_phonemes_json TEXT NOT NULL DEFAULT '[]',
    expected_patterns_json TEXT NOT NULL DEFAULT '[]',
    active INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (client_id) REFERENCES pediatric_clients(id)
);

CREATE TABLE IF NOT EXISTS clinical_sessions (
    id TEXT PRIMARY KEY,
    client_id TEXT NOT NULL,
    exercise_id TEXT NOT NULL,
    prompt_text TEXT NOT NULL,
    model_version TEXT NOT NULL,
    status TEXT NOT NULL
        CHECK(status IN ('active', 'completed', 'cancelled')),
    started_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_at TEXT,
    FOREIGN KEY (client_id) REFERENCES pediatric_clients(id),
    FOREIGN KEY (exercise_id) REFERENCES therapy_exercises(id)
);

CREATE TABLE IF NOT EXISTS clinical_repetitions (
    id INTEGER PRIMARY KEY,
    session_id TEXT NOT NULL,
    repetition_index INTEGER NOT NULL CHECK(repetition_index > 0),
    audio_sha256 TEXT NOT NULL,
    duration_ms INTEGER NOT NULL CHECK(duration_ms >= 0),
    system_status TEXT NOT NULL
        CHECK(system_status IN ('REVIEW_REQUIRED', 'RETRY')),
    observations_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(session_id, repetition_index),
    FOREIGN KEY (session_id) REFERENCES clinical_sessions(id)
);

CREATE TABLE IF NOT EXISTS clinician_reviews (
    id INTEGER PRIMARY KEY,
    repetition_id INTEGER NOT NULL,
    reviewer_id TEXT NOT NULL,
    verdict TEXT NOT NULL
        CHECK(verdict IN ('accepted', 'speech_error', 'retry')),
    observations_json TEXT NOT NULL DEFAULT '{}',
    notes TEXT NOT NULL DEFAULT '',
    reviewed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (repetition_id) REFERENCES clinical_repetitions(id)
);

CREATE TABLE IF NOT EXISTS clinical_audit_events (
    id INTEGER PRIMARY KEY,
    actor_id TEXT NOT NULL,
    action TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    payload_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_clinical_session_client
ON clinical_sessions(client_id, started_at);

CREATE INDEX IF NOT EXISTS idx_clinical_repetition_session
ON clinical_repetitions(session_id, repetition_index);

CREATE INDEX IF NOT EXISTS idx_clinical_review_repetition
ON clinician_reviews(repetition_id, reviewed_at);

CREATE TABLE IF NOT EXISTS sentence_benchmarks (
    id INTEGER PRIMARY KEY,
    sentence_id TEXT NOT NULL,
    text TEXT NOT NULL,
    language TEXT NOT NULL,
    locale TEXT NOT NULL,
    units_json TEXT NOT NULL,
    reference_blob BLOB NOT NULL,
    feature_dimension INTEGER NOT NULL,
    error_threshold REAL NOT NULL CHECK(error_threshold > 0),
    final_threshold REAL NOT NULL CHECK(final_threshold > 0),
    version TEXT NOT NULL,
    active INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(sentence_id, version)
);

CREATE INDEX IF NOT EXISTS idx_sentence_benchmark_lookup
ON sentence_benchmarks(sentence_id, active);

CREATE TABLE IF NOT EXISTS sentence_attempts (
    id INTEGER PRIMARY KEY,
    session_id TEXT NOT NULL,
    benchmark_id INTEGER NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('CORRECT', 'INCORRECT', 'RETRY')),
    distance REAL,
    errors_json TEXT NOT NULL DEFAULT '[]',
    model_version TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (benchmark_id) REFERENCES sentence_benchmarks(id)
);
"""


def _table_columns(database: sqlite3.Connection, table: str) -> set[str]:
    return {
        str(row["name"])
        for row in database.execute(f'PRAGMA table_info("{table}")').fetchall()
    }


def _next_legacy_name(database: sqlite3.Connection, table: str) -> str:
    base = f"legacy_{table}"
    candidate = base
    suffix = 1
    while database.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (candidate,),
    ).fetchone():
        suffix += 1
        candidate = f"{base}_{suffix}"
    return candidate


def _archive_incompatible_legacy_schema(database: sqlite3.Connection) -> None:
    """Preserve tables from the pre-ADVX prototype before creating v3 tables."""
    required_benchmark_columns = {
        "id",
        "language",
        "word",
        "expected_phonemes_json",
        "version",
    }
    benchmark_columns = _table_columns(database, "benchmarks")
    if not benchmark_columns or required_benchmark_columns <= benchmark_columns:
        return

    # Rename the referenced table first so SQLite updates the legacy attempts
    # foreign key, then archive attempts under its own unambiguous name.
    benchmark_archive = _next_legacy_name(database, "benchmarks")
    database.execute(
        f'ALTER TABLE "benchmarks" RENAME TO "{benchmark_archive}"'
    )

    if _table_columns(database, "attempts"):
        attempts_archive = _next_legacy_name(database, "attempts")
        database.execute(
            f'ALTER TABLE "attempts" RENAME TO "{attempts_archive}"'
        )


def main() -> None:
    database_path = get_settings().sqlite_path
    database_path.parent.mkdir(parents=True, exist_ok=True)

    with db_connection() as database:
        _archive_incompatible_legacy_schema(database)
        database.executescript(SCHEMA)
        database.execute(
            """
            INSERT OR IGNORE INTO benchmarks (
                language,
                word,
                expected_phonemes_json,
                accepted_variants_json,
                difficulty,
                version
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            ("it", "cane", '["k","a","n","e"]', "[]", 1, "1.0.0"),
        )


if __name__ == "__main__":
    main()
