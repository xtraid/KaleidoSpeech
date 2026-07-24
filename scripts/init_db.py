"""Create the SQLite schema and seed the initial benchmark."""

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
"""


def main() -> None:
    database_path = get_settings().sqlite_path
    database_path.parent.mkdir(parents=True, exist_ok=True)

    with db_connection() as database:
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
