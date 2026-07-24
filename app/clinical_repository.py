"""Minimal, auditable persistence for clinician-supervised exercises."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
import re
from uuid import uuid4

from app.database import db_connection


_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
LEVELS = {"sound", "syllable", "word", "phrase"}


def _identifier(value: str, field: str) -> str:
    if _ID.fullmatch(value) is None:
        raise ValueError(f"Invalid {field}")
    return value


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _audit(
    database, actor_id: str, action: str, entity_type: str,
    entity_id: str, payload: dict | None = None,
) -> None:
    database.execute(
        """INSERT INTO clinical_audit_events
           (actor_id, action, entity_type, entity_id, payload_json)
           VALUES (?, ?, ?, ?, ?)""",
        (actor_id, action, entity_type, entity_id,
         json.dumps(payload or {}, sort_keys=True)),
    )


def create_client(
    *, pseudonym: str, age_band: str, primary_language: str = "en",
    locale: str = "en-US", consent_status: str = "granted",
    actor_id: str,
) -> str:
    if age_band not in {"3-5", "6-8", "9-12", "13-17"}:
        raise ValueError("Unsupported age_band")
    if consent_status != "granted":
        raise ValueError("Active guardian consent is required")
    client_id = f"client_{uuid4().hex}"
    with db_connection() as database:
        database.execute("BEGIN IMMEDIATE")
        database.execute(
            """INSERT INTO pediatric_clients
               (id, pseudonym, age_band, primary_language, locale,
                consent_status, consent_recorded_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (client_id, pseudonym.strip(), age_band, primary_language,
             locale, consent_status, _now()),
        )
        _audit(database, actor_id, "client.created", "client", client_id)
        database.commit()
    return client_id


def create_exercise(
    *, client_id: str, clinician_id: str, level: str, target_text: str,
    target_phonemes: list[str] | None = None,
    expected_patterns: list[str] | None = None,
    language: str = "en", locale: str = "en-US",
) -> str:
    _identifier(client_id, "client_id")
    _identifier(clinician_id, "clinician_id")
    if level not in LEVELS:
        raise ValueError("Unsupported exercise level")
    if not target_text.strip():
        raise ValueError("target_text cannot be empty")
    exercise_id = f"exercise_{uuid4().hex}"
    with db_connection() as database:
        database.execute("BEGIN IMMEDIATE")
        consent = database.execute(
            "SELECT consent_status FROM pediatric_clients WHERE id = ?",
            (client_id,),
        ).fetchone()
        if consent is None or consent["consent_status"] != "granted":
            raise PermissionError("Active guardian consent is required")
        database.execute(
            """INSERT INTO therapy_exercises
               (id, client_id, clinician_id, level, language, locale,
                target_text, target_phonemes_json, expected_patterns_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (exercise_id, client_id, clinician_id, level, language, locale,
             target_text.strip(), json.dumps(target_phonemes or []),
             json.dumps(expected_patterns or [])),
        )
        _audit(database, clinician_id, "exercise.created", "exercise",
               exercise_id, {"level": level})
        database.commit()
    return exercise_id


def start_session(
    *, client_id: str, exercise_id: str, prompt_text: str,
    actor_id: str, model_version: str,
) -> str:
    session_id = f"session_{uuid4().hex}"
    with db_connection() as database:
        database.execute("BEGIN IMMEDIATE")
        row = database.execute(
            """SELECT client.consent_status
               FROM therapy_exercises exercise
               JOIN pediatric_clients client ON client.id = exercise.client_id
               WHERE exercise.id = ? AND exercise.client_id = ?
                 AND exercise.active = 1""",
            (exercise_id, client_id),
        ).fetchone()
        if row is None or row["consent_status"] != "granted":
            raise PermissionError("Exercise unavailable or consent withdrawn")
        database.execute(
            """INSERT INTO clinical_sessions
               (id, client_id, exercise_id, prompt_text, model_version, status)
               VALUES (?, ?, ?, ?, ?, 'active')""",
            (session_id, client_id, exercise_id, prompt_text, model_version),
        )
        _audit(database, actor_id, "session.started", "session", session_id)
        database.commit()
    return session_id


def save_repetition(
    *, session_id: str, repetition_index: int, pcm_s16le: bytes,
    duration_ms: int, system_status: str,
    observations: dict, actor_id: str = "system",
) -> int:
    if system_status not in {"REVIEW_REQUIRED", "RETRY"}:
        raise ValueError("The system cannot make a clinical verdict")
    digest = sha256(pcm_s16le).hexdigest()
    with db_connection() as database:
        database.execute("BEGIN IMMEDIATE")
        cursor = database.execute(
            """INSERT INTO clinical_repetitions
               (session_id, repetition_index, audio_sha256, duration_ms,
                system_status, observations_json)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (session_id, repetition_index, digest, duration_ms, system_status,
             json.dumps(observations, sort_keys=True)),
        )
        repetition_id = int(cursor.lastrowid)
        _audit(database, actor_id, "repetition.recorded", "repetition",
               str(repetition_id), {"status": system_status})
        database.commit()
    return repetition_id


def review_repetition(
    *, repetition_id: int, reviewer_id: str, verdict: str,
    observations: dict | None = None, notes: str = "",
) -> int:
    if verdict not in {"accepted", "speech_error", "retry"}:
        raise ValueError("Invalid clinician verdict")
    with db_connection() as database:
        database.execute("BEGIN IMMEDIATE")
        cursor = database.execute(
            """INSERT INTO clinician_reviews
               (repetition_id, reviewer_id, verdict, observations_json, notes)
               VALUES (?, ?, ?, ?, ?)""",
            (repetition_id, reviewer_id, verdict,
             json.dumps(observations or {}, sort_keys=True), notes),
        )
        review_id = int(cursor.lastrowid)
        _audit(database, reviewer_id, "repetition.reviewed", "repetition",
               str(repetition_id), {"verdict": verdict})
        database.commit()
    return review_id
