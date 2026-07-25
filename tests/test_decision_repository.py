import json

from app.config import Settings
from app.decision_repository import (
    annotate_decision,
    list_unreviewed_decisions,
    save_decision_log,
)
from app.database import db_connection


def test_decision_log_can_be_annotated(tmp_path, monkeypatch) -> None:
    settings = Settings(sqlite_path=tmp_path / "decisions.sqlite3")
    monkeypatch.setattr("app.database.get_settings", lambda: settings)

    save_decision_log(
        attempt_id="attempt_1",
        session_id="session_1",
        result={
            "status": "UNDECIDABLE",
            "reason_codes": ["WORD_IDENTITY_UNCERTAIN"],
            "target_word": "bird",
            "word_identity_confidence": 0.4,
            "versions": {"decision": "v1"},
        },
    )
    queue = list_unreviewed_decisions()
    assert queue[0]["attempt_id"] == "attempt_1"
    assert annotate_decision(
        "attempt_1", ground_truth="INCORRECT", reviewer_id="reviewer_1"
    )

    with db_connection() as database:
        row = database.execute(
            "SELECT * FROM decision_logs WHERE attempt_id = 'attempt_1'"
        ).fetchone()
    assert row["ground_truth"] == "INCORRECT"
    assert json.loads(row["signals_json"])["word_identity_confidence"] == 0.4
    assert list_unreviewed_decisions() == []
