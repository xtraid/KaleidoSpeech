from dataclasses import dataclass
import json

import pytest

from app.audio_producer import split_pcm
from app.benchmark_repository import find_benchmark, save_attempt
from app.config import get_settings
from app.pronunciation_engine import PronunciationResult
from scripts.init_db import main as initialize_database


def test_canonical_audio_frame_geometry() -> None:
    settings = get_settings()

    assert settings.samples_per_frame == 640
    assert settings.bytes_per_frame == 1280
    assert len(split_pcm(bytes(16_000 * 2), settings.samples_per_frame)) == 25


def test_split_pcm_drops_partial_tail() -> None:
    frames = split_pcm(bytes(1281), 640)

    assert len(frames) == 1
    assert len(frames[0]) == 1280


def test_sqlite_benchmark_and_attempt_round_trip(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("SQLITE_PATH", str(tmp_path / "test.sqlite3"))
    get_settings.cache_clear()
    try:
        initialize_database()
        benchmark = find_benchmark("it", "CANE")
        assert benchmark is not None
        assert benchmark.expected_phonemes == ["k", "a", "n", "e"]

        result = PronunciationResult(
            detected_phonemes=["k", "a", "n", "e"],
            phoneme_confidences=[0.9, 0.9, 0.9, 0.9],
            overall_confidence=0.9,
            score=0.95,
            engine_version="test-1",
        )
        attempt = save_attempt("session-test", benchmark.id, result)

        assert attempt.benchmark_id == benchmark.id
        assert attempt.detected_phonemes == result.detected_phonemes
        assert attempt.score == 0.95
    finally:
        get_settings.cache_clear()


@dataclass
class _Pipeline:
    calls: list[tuple]

    def hset(self, key, mapping):
        self.calls.append(("hset", key, mapping))
        return self

    def expire(self, key, ttl):
        self.calls.append(("expire", key, ttl))
        return self

    def execute(self):
        self.calls.append(("execute",))
        return []


class _Redis:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def xadd(self, key, fields, **options):
        self.calls.append(("xadd", key, fields, options))
        return b"1-0"

    def pipeline(self):
        return _Pipeline(self.calls)


def test_redis_frame_metadata_stream_limit_and_session_ttl(monkeypatch) -> None:
    from app import redis_bus

    fake = _Redis()
    monkeypatch.setattr(redis_bus, "client", fake)
    settings = get_settings()

    message_id = redis_bus.publish_audio_frame(
        "session-test",
        sequence=7,
        captured_at_ns=123,
        pcm=bytes(settings.bytes_per_frame),
    )
    redis_bus.update_session_state("session-test", {"phase": "listening"})

    assert message_id == b"1-0"
    xadd = fake.calls[0]
    assert xadd[1] == "audio:session-test"
    assert xadd[2][b"sequence"] == b"7"
    assert xadd[2][b"captured_at_ns"] == b"123"
    assert xadd[2][b"sample_rate"] == b"16000"
    assert xadd[2][b"frame_ms"] == b"40"
    assert xadd[2][b"pcm_s16le"] == bytes(settings.bytes_per_frame)
    assert xadd[3]["maxlen"] == settings.max_stream_length
    assert (
        "expire",
        "session:session-test:state",
        settings.session_ttl_seconds,
    ) in fake.calls

    hset = next(call for call in fake.calls if call[0] == "hset")
    assert json.loads(hset[2]["phase"]) == "listening"


def test_word_window_requires_real_contiguous_redis_frames() -> None:
    from app.worker import assemble_word_window

    settings = get_settings()

    def record(sequence: int) -> dict[bytes, bytes]:
        return {
            b"sequence": str(sequence).encode(),
            b"captured_at_ns": str(123 + sequence).encode(),
            b"sample_rate": str(settings.sample_rate).encode(),
            b"frame_ms": str(settings.frame_ms).encode(),
            b"pcm_s16le": bytes([sequence]) * settings.bytes_per_frame,
        }

    records = [record(4), record(5)]
    assert assemble_word_window(records) == (
        records[0][b"pcm_s16le"] + records[1][b"pcm_s16le"]
    )

    records[1][b"sequence"] = b"6"
    with pytest.raises(ValueError, match="not contiguous"):
        assemble_word_window(records)
