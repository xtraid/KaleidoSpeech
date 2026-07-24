"""Black-box contract and lightweight fuzz tests for database adapters."""

from __future__ import annotations

from dataclasses import dataclass
import json
import random
import sqlite3
import string
from typing import Any

import pytest

from app.benchmark_repository import find_benchmark, save_attempt
from app.config import get_settings
from app.database import db_connection
from app.redis_bus import (
    publish_audio_frame,
    publish_ui_event,
    update_session_state,
)
from scripts.init_db import main as initialize_database


@dataclass(frozen=True)
class Result:
    detected_phonemes: list[str]
    overall_confidence: float
    score: float | None
    engine_version: str


class RedisRecorder:
    """Minimal Redis boundary double recording commands sent by the adapter."""

    def __init__(self) -> None:
        self.commands: list[tuple[Any, ...]] = []

    def xadd(
        self,
        key: str,
        fields: dict[bytes, bytes],
        **options: Any,
    ) -> bytes:
        self.commands.append(("xadd", key, fields, options))
        return b"42-0"

    def pipeline(self) -> RedisPipelineRecorder:
        return RedisPipelineRecorder(self.commands)


class RedisPipelineRecorder:
    def __init__(self, commands: list[tuple[Any, ...]]) -> None:
        self.commands = commands

    def hset(self, key: str, mapping: dict[str, str]) -> RedisPipelineRecorder:
        self.commands.append(("hset", key, mapping))
        return self

    def expire(self, key: str, ttl: int) -> RedisPipelineRecorder:
        self.commands.append(("expire", key, ttl))
        return self

    def execute(self) -> list[object]:
        self.commands.append(("execute",))
        return []


@pytest.fixture
def isolated_sqlite(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("SQLITE_PATH", str(tmp_path / "database.sqlite3"))
    get_settings.cache_clear()
    initialize_database()
    yield
    get_settings.cache_clear()


@pytest.fixture
def redis_recorder(monkeypatch: pytest.MonkeyPatch) -> RedisRecorder:
    from app import redis_bus

    recorder = RedisRecorder()
    monkeypatch.setattr(redis_bus, "client", recorder)
    return recorder


def test_sqlite_initialization_is_idempotent(isolated_sqlite: None) -> None:
    initialize_database()
    initialize_database()

    benchmark = find_benchmark("IT", "CANE")

    assert benchmark is not None
    assert benchmark.language == "it"
    assert benchmark.word == "cane"
    with db_connection() as database:
        count = database.execute(
            "SELECT COUNT(*) FROM benchmarks WHERE language = 'it' AND word = 'cane'"
        ).fetchone()[0]
    assert count == 1


@pytest.mark.parametrize(
    ("language", "word"),
    [
        ("it' OR 1=1 --", "cane"),
        ("it", "cane' UNION SELECT 1 --"),
        ("", ""),
        ("🔥", "\x00"),
        ("a" * 2_000, "b" * 2_000),
    ],
)
def test_benchmark_lookup_treats_fuzzy_input_as_data(
    isolated_sqlite: None,
    language: str,
    word: str,
) -> None:
    assert find_benchmark(language, word) is None
    assert find_benchmark("it", "cane") is not None


def test_attempt_round_trips_unicode_and_boundary_values(
    isolated_sqlite: None,
) -> None:
    benchmark = find_benchmark("it", "cane")
    assert benchmark is not None
    cases = [
        Result([], 0.0, None, ""),
        Result(["k", "a", "ɲ", "e"], 1.0, 0.0, "引擎-🚀"),
        Result(["'" + "x" * 512, "\x00"], -0.25, 1.25, "v' OR 1=1"),
    ]

    for index, result in enumerate(cases):
        session_id = f"session-{index}-'🔥\x00"
        attempt = save_attempt(session_id, benchmark.id, result)

        assert attempt.session_id == session_id
        assert attempt.detected_phonemes == result.detected_phonemes
        assert attempt.confidence == result.overall_confidence
        assert attempt.score == result.score
        assert attempt.engine_version == result.engine_version
        assert attempt.created_at


def test_failed_attempt_does_not_poison_next_transaction(
    isolated_sqlite: None,
) -> None:
    result = Result(["k"], 0.5, 0.5, "test")

    with pytest.raises(sqlite3.IntegrityError):
        save_attempt("missing-benchmark", 999_999, result)

    benchmark = find_benchmark("it", "cane")
    assert benchmark is not None
    assert save_attempt("valid", benchmark.id, result).id > 0


def test_connection_enables_foreign_keys_and_closes_after_use(
    isolated_sqlite: None,
) -> None:
    with db_connection() as database:
        assert database.execute("PRAGMA foreign_keys").fetchone()[0] == 1

    with pytest.raises(sqlite3.ProgrammingError):
        database.execute("SELECT 1")


def test_fuzzed_valid_audio_frames_preserve_payload_and_metadata(
    redis_recorder: RedisRecorder,
) -> None:
    settings = get_settings()
    rng = random.Random(0xAD7)

    for index in range(25):
        pcm = rng.randbytes(settings.bytes_per_frame)
        sequence = rng.randint(-(2**31), 2**31)
        captured_at_ns = rng.randint(0, 2**63 - 1)
        session_id = "".join(
            rng.choice(string.ascii_letters + string.digits + ":_-")
            for _ in range(rng.randint(0, 40))
        )

        assert (
            publish_audio_frame(session_id, sequence, captured_at_ns, pcm)
            == b"42-0"
        )
        _, key, fields, options = redis_recorder.commands[-1]
        assert key == f"audio:{session_id}"
        assert fields[b"sequence"] == str(sequence).encode()
        assert fields[b"captured_at_ns"] == str(captured_at_ns).encode()
        assert fields[b"pcm_s16le"] == pcm
        assert options == {
            "maxlen": settings.max_stream_length,
            "approximate": True,
        }


@pytest.mark.parametrize("size_delta", [-2, -1, 1, 2, 1_000])
def test_audio_frame_rejects_every_nearby_invalid_size(
    redis_recorder: RedisRecorder,
    size_delta: int,
) -> None:
    expected = get_settings().bytes_per_frame

    with pytest.raises(ValueError, match=rf"Expected {expected} PCM bytes"):
        publish_audio_frame("session", 0, 0, bytes(expected + size_delta))

    assert redis_recorder.commands == []


def test_fuzzed_session_state_is_json_encoded_and_expires(
    redis_recorder: RedisRecorder,
) -> None:
    state = {
        "unicode-🔥": {"nested": [None, True, 17, "音声"]},
        "quotes'\"": "value\x00with-control",
        "long": "x" * 2_000,
    }

    update_session_state("session:odd-'🔥", state)

    hset = redis_recorder.commands[0]
    assert hset[0] == "hset"
    assert hset[1] == "session:session:odd-'🔥:state"
    assert {key: json.loads(value) for key, value in hset[2].items()} == state
    assert redis_recorder.commands[1] == (
        "expire",
        "session:session:odd-'🔥:state",
        get_settings().session_ttl_seconds,
    )
    assert redis_recorder.commands[2] == ("execute",)


def test_ui_event_round_trips_as_utf8_json(
    redis_recorder: RedisRecorder,
) -> None:
    event = {
        "type": "résultat",
        "payload": ["ɲ", {"score": 0.875, "message": "成功 🔥"}],
    }

    assert publish_ui_event("用户", event) == b"42-0"

    _, key, fields, options = redis_recorder.commands[0]
    assert key == "session:用户:events"
    assert json.loads(fields[b"json"].decode("utf-8")) == event
    assert options == {"maxlen": 500, "approximate": True}
