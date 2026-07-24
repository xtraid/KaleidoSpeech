import json

import pytest

from app.clinical_repository import (
    create_client,
    create_exercise,
    review_repetition,
    save_repetition,
    start_session,
)
from app.config import get_settings
from app.database import db_connection
from scripts.init_db import main as initialize_database


@pytest.fixture
def clinical_db(tmp_path, monkeypatch):
    monkeypatch.setenv("SQLITE_PATH", str(tmp_path / "clinical.sqlite3"))
    get_settings.cache_clear()
    initialize_database()
    yield
    get_settings.cache_clear()


def test_clinical_workflow_is_consent_gated_and_auditable(clinical_db):
    client_id = create_client(
        pseudonym="child-01", age_band="6-8", actor_id="clinician_1"
    )
    exercise_id = create_exercise(
        client_id=client_id,
        clinician_id="clinician_1",
        level="phrase",
        target_text="The happy bird will learn.",
        target_phonemes=["b", "ɝ", "d"],
    )
    session_id = start_session(
        client_id=client_id,
        exercise_id=exercise_id,
        prompt_text="The happy bird will learn.",
        actor_id="clinician_1",
        model_version="test-v1",
    )
    audio = b"\x01\x02" * 640
    repetition_id = save_repetition(
        session_id=session_id,
        repetition_index=1,
        pcm_s16le=audio,
        duration_ms=40,
        system_status="REVIEW_REQUIRED",
        observations={"voice": True},
    )
    review_id = review_repetition(
        repetition_id=repetition_id,
        reviewer_id="clinician_1",
        verdict="speech_error",
        observations={"pattern": "final consonant deletion"},
    )

    with db_connection() as database:
        repetition = database.execute(
            "SELECT * FROM clinical_repetitions WHERE id = ?",
            (repetition_id,),
        ).fetchone()
        audits = database.execute(
            "SELECT action FROM clinical_audit_events ORDER BY id"
        ).fetchall()
    assert review_id > 0
    assert repetition["audio_sha256"] != audio.hex()
    assert json.loads(repetition["observations_json"]) == {"voice": True}
    assert [row["action"] for row in audits] == [
        "client.created", "exercise.created", "session.started",
        "repetition.recorded", "repetition.reviewed",
    ]


def test_system_cannot_persist_correct_or_incorrect_verdict(clinical_db):
    with pytest.raises(ValueError, match="clinical verdict"):
        save_repetition(
            session_id="session_missing",
            repetition_index=1,
            pcm_s16le=b"",
            duration_ms=0,
            system_status="CORRECT",
            observations={},
        )


def test_client_creation_requires_consent(clinical_db):
    with pytest.raises(ValueError, match="consent"):
        create_client(
            pseudonym="child-02", age_band="6-8",
            consent_status="withdrawn", actor_id="clinician_1",
        )
