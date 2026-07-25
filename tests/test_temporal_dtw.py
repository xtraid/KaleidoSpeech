"""Contracts for temporal features, DTW artifacts, and contrastive decisions."""

from __future__ import annotations

import math

import numpy as np
import pytest

from app.acoustic_repository import store_recording
from app.config import get_settings
from app.decision import DecisionStatus
from app.temporal_benchmark import (
    _subsequence_distance_to_prototypes,
    _contrastive_decision,
    build_temporal_benchmarks,
)
from app.temporal_features import (
    FEATURE_DIMENSION,
    TEMPORAL_DISTANCE_VERSION,
    TEMPORAL_EXTRACTOR_VERSION,
    dtw_distance,
    extract_temporal_features,
)
from app.temporal_repository import (
    TemporalBenchmark,
    find_temporal_benchmark,
    list_temporal_recordings,
    store_temporal_feature,
)
from scripts.init_db import main as initialize_database


def _tone(frequency: float, seconds: float = 0.6) -> bytes:
    timeline = np.arange(round(16_000 * seconds)) / 16_000
    samples = np.rint(np.sin(2 * math.pi * frequency * timeline) * 10_000)
    return samples.astype("<i2").tobytes()


def test_temporal_features_and_dtw_are_deterministic() -> None:
    first = extract_temporal_features(_tone(220)).values
    second = extract_temporal_features(_tone(220)).values
    different = extract_temporal_features(_tone(660)).values

    np.testing.assert_array_equal(first, second)
    assert first.ndim == 2
    assert first.shape[1] == FEATURE_DIMENSION
    assert np.isfinite(first).all()
    assert dtw_distance(first, first) == pytest.approx(0.0, abs=1e-12)
    assert dtw_distance(first, second) < dtw_distance(first, different)
    assert dtw_distance(first, different) == pytest.approx(
        dtw_distance(different, first),
        abs=1e-12,
    )


def test_subsequence_dtw_finds_a_word_inside_a_sentence() -> None:
    rng = np.random.default_rng(7)
    prototype = rng.normal(size=(12, FEATURE_DIMENSION)).astype(np.float32)
    sentence = np.concatenate(
        (
            rng.normal(size=(18, FEATURE_DIMENSION)),
            prototype,
            rng.normal(size=(20, FEATURE_DIMENSION)),
        ),
        axis=0,
    ).astype(np.float32)

    distance = _subsequence_distance_to_prototypes(sentence, (prototype,))

    assert distance == pytest.approx(0.0, abs=1e-7)


def _benchmark() -> TemporalBenchmark:
    prototype = np.zeros((5, FEATURE_DIMENSION), dtype=np.float32)
    return TemporalBenchmark(
        id=1,
        word="bird",
        language="en",
        locale="en-US",
        task="word_pronunciation",
        prototype_recording_ids=(1,),
        prototypes=(prototype,),
        accept_threshold=0.2,
        reject_threshold=0.5,
        margin_threshold=0.1,
        distance_statistics={},
        sample_count=10,
        speaker_count=10,
        extractor_version=TEMPORAL_EXTRACTOR_VERSION,
        distance_version=TEMPORAL_DISTANCE_VERSION,
        decision_version="test",
        version="test",
    )


def test_contrastive_decision_requires_both_distance_and_margin() -> None:
    benchmark = _benchmark()

    accepted = _contrastive_decision(
        benchmark,
        {"bird": 0.1, "learn": 0.4},
        cleaning_status="accept",
    )
    ambiguous = _contrastive_decision(
        benchmark,
        {"bird": 0.1, "learn": 0.15},
        cleaning_status="accept",
    )
    wrong = _contrastive_decision(
        benchmark,
        {"bird": 0.3, "learn": 0.1},
        cleaning_status="accept",
    )
    retry = _contrastive_decision(
        benchmark,
        {"bird": 0.1, "learn": 0.4},
        cleaning_status="retry_recommended",
    )

    assert accepted.status is DecisionStatus.CORRECT
    assert ambiguous.status is DecisionStatus.UNDECIDABLE
    assert wrong.status is DecisionStatus.INCORRECT
    assert retry.status is DecisionStatus.RETRY


def test_temporal_artifacts_build_and_round_trip(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("SQLITE_PATH", str(tmp_path / "temporal.sqlite3"))
    get_settings.cache_clear()
    try:
        initialize_database()
        for word_index, word in enumerate(("bird", "learn")):
            for split, count in (("train", 8), ("validation", 4), ("test", 2)):
                for index in range(count):
                    recording_id = store_recording(
                        word=word,
                        language="en",
                        locale="en-US",
                        speaker_hash=f"{word}-{split}-{index}",
                        take_index=0,
                        source_hash=f"{word}-{split}-{index}",
                        source_member=f"fixture/{word}/{index}.wav",
                        dataset_split=split,
                        duration_ms=600,
                        quality_status="ACCEPT",
                        quality_reasons=[],
                        feature=np.zeros(26),
                        cleaning_version="test",
                        extractor_version="test",
                    )
                    base = word_index * 3.0
                    sequence = np.full(
                        (20 + index, FEATURE_DIMENSION),
                        base,
                        dtype=np.float32,
                    )
                    sequence[:, index % FEATURE_DIMENSION] += np.linspace(
                        0,
                        0.2,
                        len(sequence),
                    )
                    store_temporal_feature(
                        recording_id,
                        sequence,
                        extractor_version=TEMPORAL_EXTRACTOR_VERSION,
                        distance_version=TEMPORAL_DISTANCE_VERSION,
                    )

        benchmarks = build_temporal_benchmarks(version="test-dtw")

        assert [item.word for item in benchmarks] == ["bird", "learn"]
        bird = find_temporal_benchmark("BIRD")
        assert bird is not None
        assert bird.accept_threshold < bird.reject_threshold
        assert bird.margin_threshold >= 0
        assert len(bird.prototypes) == 4
        assert all(
            sequence.shape[1] == FEATURE_DIMENSION
            for sequence in bird.prototypes
        )
        assert len(list_temporal_recordings(dataset_split="test")) == 4
    finally:
        get_settings.cache_clear()
