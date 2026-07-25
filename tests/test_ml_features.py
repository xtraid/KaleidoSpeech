from app.ml_features import FEATURE_NAMES, extract_decision_features


def test_ml_feature_vector_has_stable_order_and_aggregates() -> None:
    vector = extract_decision_features(
        {
            "word_identity_confidence": 0.9,
            "pronunciation_score": 0.4,
            "duration_ms": 500,
            "voice_frame_ratio": 0.8,
            "quality": {"snr_db": 20},
            "phones": [
                {"score": 0.2, "confidence": 0.7},
                {"score": 0.6, "confidence": 0.9},
            ],
        }
    )

    assert len(vector.values) == len(FEATURE_NAMES)
    assert vector.values[2:6] == (0.4, 0.2, 0.8, 0.7)
