from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


FIXTURES = Path(__file__).parent / "fixtures" / "voicebank_demand"
TRACK_IDS = ("p232_036", "p232_075", "p232_080", "p232_145")


@pytest.fixture(params=TRACK_IDS, ids=TRACK_IDS)
def benchmark_case(request: pytest.FixtureRequest) -> dict[str, object]:
    """A real VoiceBank-DEMAND noisy/clean/transcript benchmark case."""
    track_id = str(request.param)
    noisy_bytes = (FIXTURES / "redis" / f"{track_id}.f32").read_bytes()
    clean_bytes = (FIXTURES / "reference" / f"{track_id}.f32").read_bytes()
    transcript = (
        FIXTURES / "transcripts" / f"{track_id}.txt"
    ).read_text(encoding="utf-8").strip()
    return {
        "id": track_id,
        "redis_record": {b"audio": noisy_bytes},
        "clean_reference": np.frombuffer(clean_bytes, dtype=np.float32),
        "transcript": transcript,
    }


@pytest.fixture
def redis_silence_record() -> dict[bytes, bytes]:
    silence = np.zeros(16_000, dtype=np.float32)
    return {b"audio": silence.tobytes()}
