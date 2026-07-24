from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.audio_producer import split_pcm
from app.config import get_settings


FIXTURES = Path(__file__).parent / "fixtures" / "voicebank_demand"
TRACK_IDS = ("p232_036", "p232_075", "p232_080", "p232_145")


@pytest.fixture(params=TRACK_IDS, ids=TRACK_IDS)
def benchmark_case(request: pytest.FixtureRequest) -> dict[str, object]:
    """A real VoiceBank-DEMAND noisy/clean/transcript benchmark case."""
    track_id = str(request.param)
    noisy_bytes = (FIXTURES / "redis" / f"{track_id}.f32").read_bytes()
    noisy_float = np.frombuffer(noisy_bytes, dtype="<f4")
    noisy_pcm = np.clip(
        np.rint(noisy_float * 32768.0),
        -32768,
        32767,
    ).astype("<i2").tobytes()
    settings = get_settings()
    frames = split_pcm(noisy_pcm, settings.samples_per_frame)
    redis_records = [
        {
            b"sequence": str(sequence).encode(),
            b"captured_at_ns": str(1_000_000_000 + sequence).encode(),
            b"sample_rate": str(settings.sample_rate).encode(),
            b"frame_ms": str(settings.frame_ms).encode(),
            b"pcm_s16le": frame,
        }
        for sequence, frame in enumerate(frames)
    ]
    clean_bytes = (FIXTURES / "reference" / f"{track_id}.f32").read_bytes()
    transcript = (
        FIXTURES / "transcripts" / f"{track_id}.txt"
    ).read_text(encoding="utf-8").strip()
    return {
        "id": track_id,
        "redis_records": redis_records,
        "input_pcm_s16le": b"".join(frames),
        "clean_reference": np.frombuffer(clean_bytes, dtype=np.float32),
        "transcript": transcript,
    }


@pytest.fixture
def redis_silence_records() -> list[dict[bytes, bytes]]:
    settings = get_settings()
    silence = bytes(settings.bytes_per_frame)
    return [
        {
            b"sequence": str(sequence).encode(),
            b"captured_at_ns": str(1_000_000_000 + sequence).encode(),
            b"sample_rate": str(settings.sample_rate).encode(),
            b"frame_ms": str(settings.frame_ms).encode(),
            b"pcm_s16le": silence,
        }
        for sequence in range(25)
    ]
