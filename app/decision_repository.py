"""Persist versioned decision signals for audit and later supervised learning."""

from __future__ import annotations

import json
from typing import Any

from app.database import db_connection


_SCHEMA = """
CREATE TABLE IF NOT EXISTS decision_logs (
    id INTEGER PRIMARY KEY,
    attempt_id TEXT NOT NULL UNIQUE,
    session_id TEXT NOT NULL,
    target_word TEXT,
    decision_status TEXT NOT NULL,
    reason_codes_json TEXT NOT NULL,
    signals_json TEXT NOT NULL,
    versions_json TEXT NOT NULL,
    ground_truth TEXT,
    reviewer_id TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    reviewed_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_decision_log_session
ON decision_logs(session_id, created_at);
"""


def save_decision_log(
    *,
    attempt_id: str,
    session_id: str,
    result: dict,
) -> None:
    """Store no raw audio or direct identifiers; only decision evidence."""
    excluded = {"status", "reason_codes", "versions", "target_word"}
    signals = {key: value for key, value in result.items() if key not in excluded}
    with db_connection() as database:
        database.executescript(_SCHEMA)
        database.execute(
            """INSERT INTO decision_logs
               (attempt_id, session_id, target_word, decision_status,
                reason_codes_json, signals_json, versions_json)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                attempt_id,
                session_id,
                result.get("target_word"),
                result["status"],
                json.dumps(result.get("reason_codes", []), sort_keys=True),
                json.dumps(signals, sort_keys=True),
                json.dumps(result.get("versions", {}), sort_keys=True),
            ),
        )


def annotate_decision(
    attempt_id: str, *, ground_truth: str, reviewer_id: str
) -> bool:
    if ground_truth not in {"CORRECT", "INCORRECT", "RETRY", "UNDECIDABLE"}:
        raise ValueError("Unsupported ground_truth")
    with db_connection() as database:
        database.executescript(_SCHEMA)
        cursor = database.execute(
            """UPDATE decision_logs
               SET ground_truth = ?, reviewer_id = ?, reviewed_at = CURRENT_TIMESTAMP
               WHERE attempt_id = ?""",
            (ground_truth, reviewer_id, attempt_id),
        )
        return cursor.rowcount == 1


def list_unreviewed_decisions(*, limit: int = 50) -> list[dict[str, Any]]:
    if not 1 <= limit <= 200:
        raise ValueError("limit must be between 1 and 200")
    with db_connection() as database:
        database.executescript(_SCHEMA)
        rows = database.execute(
            """SELECT attempt_id, session_id, target_word, decision_status,
                      reason_codes_json, signals_json, versions_json, created_at
               FROM decision_logs
               WHERE ground_truth IS NULL
               ORDER BY created_at, id
               LIMIT ?""",
            (limit,),
        ).fetchall()
    return [
        {
            "attempt_id": row["attempt_id"],
            "session_id": row["session_id"],
            "target_word": row["target_word"],
            "decision_status": row["decision_status"],
            "reason_codes": json.loads(row["reason_codes_json"]),
            "signals": json.loads(row["signals_json"]),
            "versions": json.loads(row["versions_json"]),
            "created_at": row["created_at"],
        }
        for row in rows
    ]
