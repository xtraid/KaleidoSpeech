"""Black-box acceptance tests: Redis payload in, solver payload out."""

from __future__ import annotations

import numpy as np
import pytest

from cleaning_contract import clean_redis_record, load_cleaner


TARGET_SAMPLE_RATE = 16_000
PEAK_LIMIT = 10 ** (-1.0 / 20.0)


def solver_samples(cleaned_audio: bytes) -> np.ndarray:
    """Decode exactly as the current solver does."""
    assert isinstance(cleaned_audio, bytes)
    assert len(cleaned_audio) % np.dtype(np.float32).itemsize == 0
    return np.frombuffer(cleaned_audio, dtype=np.float32)


def test_redis_track_becomes_canonical_solver_audio(
    benchmark_case: dict[str, object],
) -> None:
    redis_record = benchmark_case["redis_record"]
    assert isinstance(redis_record, dict)
    cleaned_audio = clean_redis_record(redis_record)
    samples = solver_samples(cleaned_audio)
    input_samples = solver_samples(redis_record[b"audio"])

    assert samples.ndim == 1
    assert samples.size > 0
    assert np.isfinite(samples).all()
    assert np.max(np.abs(samples)) <= PEAK_LIMIT + 1e-6
    assert abs(float(np.mean(samples))) < 1e-3

    duration_s = samples.size / TARGET_SAMPLE_RATE
    assert 0.35 <= duration_s <= 15.0
    assert samples.size <= input_samples.size


def test_cleaning_is_deterministic(
    benchmark_case: dict[str, object],
) -> None:
    redis_record = benchmark_case["redis_record"]
    assert isinstance(redis_record, dict)
    first = clean_redis_record(redis_record)
    second = clean_redis_record(redis_record)

    assert first == second


def test_benchmark_has_clean_reference_and_official_transcript(
    benchmark_case: dict[str, object],
) -> None:
    reference = benchmark_case["clean_reference"]
    transcript = benchmark_case["transcript"]

    assert isinstance(reference, np.ndarray)
    assert reference.dtype == np.float32
    assert reference.size > 0
    assert np.isfinite(reference).all()
    assert isinstance(transcript, str)
    assert transcript.endswith((".", "!", "?"))


def test_silence_is_not_forwarded_as_valid_solver_audio(
    redis_silence_record: dict[bytes, bytes],
) -> None:
    cleaner = load_cleaner()
    with pytest.raises(ValueError):
        cleaner(redis_silence_record[b"audio"])
