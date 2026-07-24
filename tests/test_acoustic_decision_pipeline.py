"""Black-box contracts for the English acoustic decision prototype."""

from __future__ import annotations

from io import BytesIO
import math
import random
import wave
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pytest

from app.acoustic_features import extract_features
from app.dataset_ingest import deterministic_split, iter_zip_recordings
from app.decision import decide
from app.sentence_composer import compose_sentence


def _wav_bytes(
    *,
    frequency_hz: float = 220.0,
    sample_rate: int = 16_000,
    duration_seconds: float = 0.25,
) -> bytes:
    sample_count = round(sample_rate * duration_seconds)
    timeline = np.arange(sample_count, dtype=np.float64) / sample_rate
    samples = np.rint(
        np.sin(2.0 * math.pi * frequency_hz * timeline) * 10_000
    ).astype("<i2")
    output = BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(samples.tobytes())
    return output.getvalue()


def _recording_value(recording, name: str):
    if isinstance(recording, dict):
        return recording[name]
    return getattr(recording, name)


def _decision_status(result) -> str:
    value = result["status"] if isinstance(result, dict) else result.status
    return value.value if hasattr(value, "value") else str(value)


def _feature_values(result) -> np.ndarray:
    value = result["values"] if isinstance(result, dict) else getattr(
        result, "values", result
    )
    return np.asarray(value)


def _sentence_text(result) -> str:
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        return result["text"]
    return result.text


def test_feature_extraction_is_finite_deterministic_and_fixed_width() -> None:
    pcm = _wav_bytes()[44:]

    first = _feature_values(extract_features(pcm, sample_rate=16_000))
    second = _feature_values(extract_features(pcm, sample_rate=16_000))

    assert first.ndim in (1, 2)
    assert first.size > 0
    np.testing.assert_array_equal(first, second)
    assert np.isfinite(first).all()


def test_fuzzed_valid_pcm_always_produces_finite_features() -> None:
    rng = random.Random(0xAD7)

    for sample_count in (400, 1_600, 4_001):
        samples = np.asarray(
            [rng.randint(-32_768, 32_767) for _ in range(sample_count)],
            dtype="<i2",
        )
        features = _feature_values(
            extract_features(samples.tobytes(), sample_rate=16_000)
        )

        assert features.ndim in (1, 2)
        assert features.size > 0
        assert np.isfinite(features).all()


def test_zip_ingest_ignores_metadata_and_unsafe_members(tmp_path) -> None:
    archive = tmp_path / "fragment.zip"
    with ZipFile(archive, "w", compression=ZIP_DEFLATED) as output:
        output.writestr("1/happy/a11ce000_nohash_0.wav", _wav_bytes())
        output.writestr(
            "1/follow/b0b00000_nohash_2.wav", _wav_bytes(frequency_hz=330)
        )
        output.writestr("1/.DS_Store", b"metadata")
        output.writestr(
            "__MACOSX/1/happy/._a11ce000_nohash_0.wav", b"resource fork"
        )

    recordings = list(iter_zip_recordings([archive]))

    assert len(recordings) == 2
    assert {_recording_value(recording, "word") for recording in recordings} == {
        "happy",
        "follow",
    }
    assert len(
        {_recording_value(recording, "speaker_id") for recording in recordings}
    ) == 2


@pytest.mark.parametrize(
    "unsafe_name",
    [
        "../escaped/bad00000_nohash_0.wav",
        "/absolute/bad00000_nohash_0.wav",
        r"..\escaped\bad00000_nohash_0.wav",
    ],
)
def test_zip_ingest_rejects_path_traversal(tmp_path, unsafe_name: str) -> None:
    archive = tmp_path / "unsafe.zip"
    with ZipFile(archive, "w") as output:
        output.writestr(unsafe_name, _wav_bytes())

    with pytest.raises(ValueError, match="unsafe"):
        list(iter_zip_recordings([archive]))


def test_speaker_split_is_deterministic_and_never_depends_on_word() -> None:
    speakers = [f"speaker-{index}" for index in range(500)]

    first = [deterministic_split(speaker) for speaker in speakers]
    second = [deterministic_split(speaker) for speaker in speakers]

    assert first == second
    assert set(first) == {"train", "validation", "test"}
    assert all(
        deterministic_split(speaker) == deterministic_split(speaker)
        for speaker in speakers
    )


@pytest.mark.parametrize(
    ("distance", "quality_status", "expected"),
    [
        (0.10, "ok", "CORRECT"),
        (0.50, "ok", "UNDECIDABLE"),
        (0.90, "ok", "INCORRECT"),
        (0.10, "retry_recommended", "RETRY"),
    ],
)
def test_decision_exposes_all_four_states(
    distance: float,
    quality_status: str,
    expected: str,
) -> None:
    result = decide(
        distance=distance,
        accept_threshold=0.25,
        reject_threshold=0.75,
        cleaning_status=quality_status,
    )

    assert _decision_status(result).upper() == expected


def test_retry_takes_precedence_over_an_accepting_distance() -> None:
    result = decide(
        distance=0.0,
        accept_threshold=0.25,
        reject_threshold=0.75,
        cleaning_status="retry_recommended",
    )

    assert _decision_status(result).upper() == "RETRY"


def test_decision_boundaries_reason_codes_and_version_are_explicit() -> None:
    accepted = decide(
        distance=0.25,
        accept_threshold=0.25,
        reject_threshold=0.75,
    )
    rejected = decide(
        distance=0.75,
        accept_threshold=0.25,
        reject_threshold=0.75,
    )

    assert _decision_status(accepted).upper() == "CORRECT"
    assert _decision_status(rejected).upper() == "INCORRECT"
    for result in (accepted, rejected):
        reasons = (
            result["reason_codes"]
            if isinstance(result, dict)
            else result.reason_codes
        )
        version = (
            result["decision_version"]
            if isinstance(result, dict)
            else result.decision_version
        )
        assert reasons
        assert version


def test_sentence_composition_is_seeded_and_contains_every_target() -> None:
    targets = ["happy", "bird", "learn"]

    first = compose_sentence(targets, seed=137)
    second = compose_sentence(targets, seed=137)
    text = _sentence_text(first)
    tokens = {
        token.strip(".,!?;:").casefold()
        for token in text.split()
    }

    assert first == second
    assert all(target in tokens for target in targets)
    assert text.endswith((".", "!", "?"))


def test_sentence_composer_rejects_empty_or_unsupported_targets() -> None:
    with pytest.raises(ValueError):
        compose_sentence([], seed=0)

    with pytest.raises(ValueError):
        compose_sentence(["happy", "../escape"], seed=0)
