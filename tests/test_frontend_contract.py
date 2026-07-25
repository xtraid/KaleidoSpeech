from app.frontend_contract import pronunciation_evaluated_event


def test_final_decision_is_flattened_for_frontend_contract() -> None:
    event = pronunciation_evaluated_event(
        attempt_id="attempt_1",
        session_id="session_1",
        result={
            "status": "CORRECT",
            "reason_codes": ["POSITIVE_PHONETIC_EVIDENCE"],
            "target_word": "bird",
            "word_identity_confidence": 0.92,
            "phones": [
                {"expected": "b", "observed": "b"},
                {"expected": "ɜː", "observed": "ɜː"},
            ],
            "versions": {"lexicon": "lex-v1", "decision": "engine-v1"},
        },
    )

    assert event["type"] == "pronunciation.evaluated"
    assert event["version"] == 1
    assert event["attempt_id"] == "attempt_1"
    assert event["confidence"] == 0.92
    assert event["score"] == 1.0
    assert event["expected_phonemes"] == ["b", "ɜː"]
    assert event["detected_phonemes"] == ["b", "ɜː"]
    assert event["engine_version"] == "engine-v1"


def test_undecidable_result_is_low_confidence_for_frontend() -> None:
    event = pronunciation_evaluated_event(
        attempt_id="attempt_2",
        session_id="session_1",
        result={"status": "UNDECIDABLE"},
    )

    assert event["score"] is None
    assert event["confidence"] == 0.0
