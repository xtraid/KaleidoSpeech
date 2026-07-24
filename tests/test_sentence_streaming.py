from __future__ import annotations

import numpy as np

from app.sentence_streaming import (
    OnlineSentenceAligner,
    SentenceBenchmark,
    SentenceStreamingSession,
    SentenceUnit,
)
from app.temporal_features import FEATURE_DIMENSION


def _benchmark() -> SentenceBenchmark:
    reference = np.zeros((8, FEATURE_DIMENSION), dtype=np.float32)
    for index in range(8):
        reference[index, index] = 1.0
    return SentenceBenchmark(
        id=1,
        sentence_id="test-1",
        text="happy bird",
        language="en",
        locale="en-US",
        reference=reference,
        units=(
            SentenceUnit("happy", 0, 400),
            SentenceUnit("bird", 400, 800),
        ),
        error_threshold=0.5,
        final_threshold=0.25,
        version="1",
    )


def test_online_alignment_reaches_end_for_matching_sequence() -> None:
    benchmark = _benchmark()
    aligner = OnlineSentenceAligner(benchmark)
    updates = aligner.push(benchmark.reference)
    assert updates[-1][0] == len(benchmark.reference) - 1
    assert aligner.final_distance() == 0.0
    assert aligner.unit_at(updates[-1][0])[1].text == "bird"


def test_online_alignment_exposes_local_error_position() -> None:
    benchmark = _benchmark()
    observed = benchmark.reference.copy()
    observed[5] = -observed[5]
    aligner = OnlineSentenceAligner(benchmark)
    updates = aligner.push(observed)
    reference_index, local_cost = updates[5]
    unit_index, unit = aligner.unit_at(reference_index)
    assert unit_index == 1
    assert unit.text == "bird"
    assert local_cost > benchmark.error_threshold


def test_stream_requires_exact_40_ms_frame() -> None:
    session = SentenceStreamingSession(_benchmark())
    try:
        session.push_frame(b"\0" * 10)
    except ValueError as error:
        assert "40 ms" in str(error)
    else:
        raise AssertionError("invalid frame accepted")


def test_voice_in_early_frames_keeps_chunk_when_last_frame_is_silent(
    monkeypatch,
) -> None:
    session = SentenceStreamingSession(_benchmark(), voice_threshold=0.01)
    monkeypatch.setattr(
        "app.sentence_streaming.extract_streaming_features",
        lambda *_args, **_kwargs: _benchmark().reference[:2],
    )
    voiced = np.full(640, 10_000, dtype="<i2").tobytes()
    silence = bytes(1_280)

    events = []
    for frame in (voiced, voiced, silence, silence, silence):
        events = session.push_frame(frame)

    alignment = next(
        event for event in events if event["type"] == "sentence.alignment"
    )
    assert alignment["chunk_voice_activity"] is True
    assert alignment["chunk_rms"] > session.voice_threshold
