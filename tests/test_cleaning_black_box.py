"""Black-box acceptance tests: Redis payload in, solver payload out."""

from __future__ import annotations

import numpy as np
import pytest

from app.benchmark_repository import find_benchmark, save_attempt
from app.config import get_settings
from app.pronunciation_engine import PronunciationResult
from cleaning_contract import clean_redis_records, load_cleaner
from app.worker import assemble_word_window
from scripts.init_db import main as initialize_database


TARGET_SAMPLE_RATE = 16_000
PEAK_LIMIT = 10 ** (-1.0 / 20.0)


def solver_samples(cleaned_audio: bytes) -> np.ndarray:
    """Decode exactly as the current solver does."""
    assert isinstance(cleaned_audio, bytes)
    assert len(cleaned_audio) % np.dtype("<i2").itemsize == 0
    return np.frombuffer(cleaned_audio, dtype="<i2")


def test_redis_track_becomes_canonical_solver_audio(
    benchmark_case: dict[str, object],
) -> None:
    redis_records = benchmark_case["redis_records"]
    assert isinstance(redis_records, list)
    cleaned_audio = clean_redis_records(redis_records)
    integer_samples = solver_samples(cleaned_audio)
    samples = integer_samples.astype(np.float32) / 32768.0
    input_samples = solver_samples(assemble_word_window(redis_records))

    assert samples.ndim == 1
    assert samples.size > 0
    assert integer_samples.dtype == np.dtype("<i2")
    assert np.isfinite(samples).all()
    assert np.max(np.abs(samples)) <= PEAK_LIMIT + 1e-6
    assert abs(float(np.mean(samples))) < 1e-3

    duration_s = samples.size / TARGET_SAMPLE_RATE
    assert 0.35 <= duration_s <= 15.0
    assert samples.size <= input_samples.size


def test_cleaning_is_deterministic(
    benchmark_case: dict[str, object],
) -> None:
    redis_records = benchmark_case["redis_records"]
    assert isinstance(redis_records, list)
    first = clean_redis_records(redis_records)
    second = clean_redis_records(redis_records)

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
    redis_silence_records: list[dict[bytes, bytes]],
) -> None:
    cleaner = load_cleaner()
    with pytest.raises(ValueError):
        cleaner(assemble_word_window(redis_silence_records))


def test_cleaned_int16_window_reaches_engine_and_sqlite(
    benchmark_case: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """Exercise the Redis -> cleaning -> engine -> SQLite boundaries."""
    redis_records = benchmark_case["redis_records"]
    assert isinstance(redis_records, list)
    cleaned_pcm = clean_redis_records(redis_records)

    monkeypatch.setenv("SQLITE_PATH", str(tmp_path / "contract.sqlite3"))
    get_settings.cache_clear()
    try:
        initialize_database()
        benchmark = find_benchmark("it", "cane")
        assert benchmark is not None

        observed: dict[str, object] = {}

        class ContractEngine:
            def evaluate(
                self,
                pcm_s16le: bytes,
                sample_rate: int,
                expected_phonemes: list[str],
                accepted_variants: list[list[str]],
            ) -> PronunciationResult:
                observed.update(
                    pcm=pcm_s16le,
                    sample_rate=sample_rate,
                    expected_phonemes=expected_phonemes,
                    accepted_variants=accepted_variants,
                )
                assert np.frombuffer(pcm_s16le, dtype="<i2").size > 0
                return PronunciationResult(
                    detected_phonemes=expected_phonemes,
                    phoneme_confidences=[0.9] * len(expected_phonemes),
                    overall_confidence=0.9,
                    score=0.95,
                    engine_version="contract-test",
                )

        result = ContractEngine().evaluate(
            cleaned_pcm,
            16_000,
            benchmark.expected_phonemes,
            benchmark.accepted_variants,
        )
        attempt = save_attempt("contract-session", benchmark.id, result)

        assert observed["pcm"] == cleaned_pcm
        assert observed["sample_rate"] == 16_000
        assert attempt.detected_phonemes == benchmark.expected_phonemes
        assert attempt.score == 0.95
    finally:
        get_settings.cache_clear()
