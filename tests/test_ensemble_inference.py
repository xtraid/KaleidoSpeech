import pytest

from app.ensemble_inference import ComponentResult, EnsembleDecisionEngine


def _phonetic_result(status: str = "CORRECT") -> dict:
    return {
        "status": status,
        "reason_codes": ["PHONETIC"],
        "target_word": "bird",
        "word_identity_confidence": 0.9,
        "pronunciation_score": 0.5,
        "phones": [{"score": 0.5, "confidence": 1.0}],
        "quality": {"acceptable": True, "snr_db": 20.0},
        "versions": {"decision": "phonetic-v1"},
        "words": [{"word": "bird", "status": status}],
    }


def test_weighted_ensemble_favors_phonetic_and_updates_ui_result() -> None:
    engine = EnsembleDecisionEngine(
        {
            "version": "ensemble-v1",
            "kind": "weighted_ensemble",
            "ensemble_weights": {
                "phonetic": 0.6,
                "dtw": 0.25,
                "single_word": 0.15,
            },
        },
        dtw_scorer=lambda _word, _pcm: ComponentResult(
            "INCORRECT", 0.0, 1.0
        ),
        single_word_scorer=lambda _word, _pcm: ComponentResult(
            "INCORRECT", 0.0, 1.0
        ),
    )

    result = engine.evaluate(
        _phonetic_result(),
        bytes(3200),
        voice_frame_ratio=0.8,
    )

    assert result["ensemble"]["weighted_score"] == pytest.approx(0.6)
    assert result["status"] == "INCORRECT"
    assert result["ensemble"]["activated_score"] == 0
    assert result["versions"]["decision"] == "ensemble-v1"
    assert result["versions"]["phonetic_decision"] == "phonetic-v1"
    assert result["words"][0]["status"] == "INCORRECT"


def test_ensemble_renormalizes_weights_when_word_models_are_missing() -> None:
    def unavailable(_word: str, _pcm: bytes) -> ComponentResult:
        raise LookupError

    engine = EnsembleDecisionEngine(
        {"version": "ensemble-v1", "kind": "weighted_ensemble"},
        dtw_scorer=unavailable,
        single_word_scorer=unavailable,
    )

    result = engine.evaluate(
        _phonetic_result(),
        bytes(3200),
        voice_frame_ratio=0.8,
    )

    assert result["ensemble"]["weighted_score"] == 1.0
    assert result["status"] == "CORRECT"
    assert not result["ensemble"]["components"]["dtw"]["available"]


def test_retry_from_phonetic_quality_cannot_be_overridden() -> None:
    positive = lambda _word, _pcm: ComponentResult("CORRECT", 1.0, 1.0)
    engine = EnsembleDecisionEngine(
        {"version": "ensemble-v1", "kind": "weighted_ensemble"},
        dtw_scorer=positive,
        single_word_scorer=positive,
    )

    result = engine.evaluate(
        _phonetic_result("RETRY"),
        bytes(3200),
        voice_frame_ratio=0.0,
    )

    assert result["status"] == "RETRY"
