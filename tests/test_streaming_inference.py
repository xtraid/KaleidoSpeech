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


def test_sentence_mode_preserves_the_ui_prompt() -> None:
    session = StreamingInferenceSession(
        ["bird"],
        expected_text="A happy bird can fly",
        sentence_accept_threshold=0.3,
        distance_provider=lambda _: {"bird": 0.1},
    )

    event = session.start_event()

    assert event["prompt"] == "A happy bird can fly"
    assert event["expected_text"] == "A happy bird can fly"
    assert event["mode"] == "sentence_pronunciation"


def test_sentence_mode_never_accepts_without_phonetic_evidence() -> None:
    session = StreamingInferenceSession(
        ["bird"],
        expected_text="A happy bird can fly",
        sentence_accept_threshold=0.3,
        distance_provider=lambda _: {"bird": 0.1, "happy": 0.8},
        config=StreamingConfig(
            background_inference=False,
            voice_rms_threshold=0.0,
        ),
    )
    for _ in range(25):
        session.push_frame(_frame())

    result = session.finish()

    assert result["status"] == "REVIEW_REQUIRED"
    assert result["reason_codes"] == ["PHONETIC_BACKEND_REQUIRED"]
    assert result["provisional"] is True


def test_sentence_phonetic_evaluator_is_authoritative() -> None:
    calls = []

    def evaluator(pcm: bytes, text: str, target: str) -> dict:
        calls.append((pcm, text, target))
        return {
            "status": "INCORRECT",
            "reason_codes": ["PHONE_ERROR"],
            "phones": [],
        }

    config = StreamingConfig(
        background_inference=False,
        voice_rms_threshold=0.0,
    )
    session = StreamingInferenceSession(
        ["bird"],
        expected_text="A happy bird can fly",
        sentence_accept_threshold=0.3,
        distance_provider=lambda _: {"bird": 0.1, "happy": 0.8},
        sentence_final_evaluator=evaluator,
        config=config,
    )
    frame = _frame()
    for _ in range(10):
        session.push_frame(frame)

    result = session.finish()

    assert result["status"] == "INCORRECT"
    assert result["provisional"] is False
    assert calls == [(frame * 10, "A happy bird can fly", "bird")]


def test_sentence_mode_rejects_out_of_vocabulary_audio_by_absolute_distance() -> None:
    session = StreamingInferenceSession(
        ["bird"],
        expected_text="A happy bird can fly",
        sentence_accept_threshold=0.3,
        distance_provider=lambda _: {"bird": 0.72, "happy": 0.9},
        config=StreamingConfig(
            background_inference=False,
            voice_rms_threshold=0.0,
        ),
    )
    for _ in range(25):
        session.push_frame(_frame())

    result = session.finish()

    assert result["status"] == "INCORRECT"
    assert result["reason_codes"] == ["TARGET_WORD_NOT_DETECTED"]


def test_sentence_progress_does_not_claim_words_from_vad_segments() -> None:
    session = StreamingInferenceSession(
        ["bird"],
        expected_text="A happy bird can fly",
        sentence_accept_threshold=0.3,
        distance_provider=lambda _: {},
        config=StreamingConfig(
            background_inference=False,
            voice_rms_threshold=0.01,
            minimum_window_ms=10_000,
        ),
    )

    first = session.push_frame(_frame())
    second = session.push_frame(_frame())
    session.push_frame(_frame(0))
    acquired = session.push_frame(_frame(0))

    assert first["current_word_index"] is None
    assert second["current_word_index"] is None
    assert acquired["word_acquired_index"] is None
    assert acquired["current_word_index"] is None
    assert all(word["status"] == "PENDING" for word in acquired["words"])


def test_sentence_stream_emits_stable_phonetic_word_updates() -> None:
    def evaluate(_pcm: bytes, _text: str, _target: str) -> dict:
        return {
            "words": [{
                "index": 0,
                "word": "bird",
                "status": "CORRECT",
                "pronunciation_score": 0.9,
                "phones": [{"end_ms": 200}],
            }]
        }

    session = StreamingInferenceSession(
        ["bird"],
        expected_text="bird",
        sentence_final_evaluator=evaluate,
        distance_provider=lambda _: {},
        config=StreamingConfig(
            background_inference=False,
            voice_rms_threshold=0.01,
            minimum_window_ms=10_000,
            phonetic_refresh_ms=80,
        ),
    )

    updates = []
    for _ in range(20):
        updates.extend(session.push_frame(_frame())["word_updates"])
        if updates:
            break

    assert updates == [{
        "index": 0,
        "word": "bird",
        "status": "CORRECT",
        "pronunciation_score": 0.9,
        "phones": [{"end_ms": 200}],
    }]


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


def test_final_phonetic_evaluator_runs_on_complete_single_word():
    calls = []

    def evaluator(pcm: bytes, target: str) -> dict:
        calls.append((pcm, target))
        return {"status": "INCORRECT", "reason_codes": ["PHONE_ERROR"]}

    config = StreamingConfig(background_inference=False, voice_rms_threshold=0.0)
    session = StreamingInferenceSession(
        ["bird"], distance_provider=lambda _: {}, final_evaluator=evaluator,
        config=config,
    )
    frame = bytes(config.bytes_per_frame)
    for _ in range(10):
        session.push_frame(frame)
    result = session.finish()

    assert result["status"] == "INCORRECT"
    assert result["provisional"] is False
    assert calls == [(frame * 10, "bird")]


def test_final_without_phonetic_backend_rejects_a_confident_wrong_word():
    session = StreamingInferenceSession(
        ["bird"],
        distance_provider=lambda _: {"bird": 0.8, "happy": 0.1},
        config=StreamingConfig(
            background_inference=False,
            voice_rms_threshold=0.0,
        ),
    )
    for _ in range(10):
        session.push_frame(_frame())

    result = session.finish()

    assert result["status"] == "INCORRECT"
    assert result["reason_codes"] == ["WRONG_WORD"]
    assert result["target_word"] == "bird"
    assert result["recognized_word"] == "happy"


def test_word_match_requires_phonetic_backend_for_correct_verdict():
    session = StreamingInferenceSession(
        ["bird"],
        distance_provider=lambda _: {"bird": 0.1, "happy": 0.8},
        config=StreamingConfig(
            background_inference=False,
            voice_rms_threshold=0.0,
        ),
    )
    for _ in range(10):
        session.push_frame(_frame())

    result = session.finish()

    assert result["status"] == "REVIEW_REQUIRED"
    assert result["reason_codes"] == ["PHONETIC_BACKEND_DISABLED"]
    assert result["recognized_word"] == "bird"
