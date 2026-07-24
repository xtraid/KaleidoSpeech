import numpy as np
import pytest

from app.streaming_inference import StreamingConfig, StreamingInferenceSession


def _frame(amplitude: float = 0.1) -> bytes:
    samples = np.full(640, round(amplitude * 32767), dtype="<i2")
    return samples.tobytes()


def test_stream_emits_one_observation_per_40ms_frame_and_refreshes_model():
    calls: list[int] = []

    def distances(pcm: bytes) -> dict[str, float]:
        calls.append(len(pcm))
        return {"bird": 0.3, "happy": 0.8}

    session = StreamingInferenceSession(
        ["bird"],
        distance_provider=distances,
        config=StreamingConfig(
            minimum_window_ms=80,
            rolling_window_ms=160,
            acoustic_refresh_ms=80,
            background_inference=False,
        ),
    )
    first = session.push_frame(_frame())
    second = session.push_frame(_frame())

    assert first["elapsed_ms"] == 40
    assert first["model_refreshed"] is False
    assert second["elapsed_ms"] == 80
    assert second["model_refreshed"] is True
    assert second["closest_benchmark_word"] == "bird"
    assert calls == [2560]


def test_stream_rejects_non_40ms_frames():
    session = StreamingInferenceSession(
        ["bird"], distance_provider=lambda _: {}
    )
    with pytest.raises(ValueError, match="1280 bytes"):
        session.push_frame(b"\0" * 100)


def test_final_is_review_required_not_automatic_clinical_verdict():
    session = StreamingInferenceSession(
        ["bird", "happy"], distance_provider=lambda _: {"bird": 0.2}
    )
    for _ in range(10):
        session.push_frame(_frame())
    final = session.finish()

    assert final["status"] == "REVIEW_REQUIRED"
    assert final["clinical_use"] is False
    assert final["observations"]["alignment_status"] == "not_available"


def test_silent_short_stream_requests_retry():
    session = StreamingInferenceSession(
        ["bird"], distance_provider=lambda _: {}
    )
    for _ in range(5):
        session.push_frame(_frame(0))
    assert session.finish()["status"] == "RETRY"
