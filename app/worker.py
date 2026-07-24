"""Redis consumer primitives and final pronunciation-result handling."""

import argparse
import json
import socket
from collections.abc import Callable
from dataclasses import asdict
from typing import Any

from redis.exceptions import ResponseError

from app.benchmark_repository import (
    Benchmark,
    find_benchmark,
    save_attempt,
)
from app.pronunciation_engine import PronunciationResult
from app.redis_bus import client, publish_ui_event, update_session_state


FrameProcessor = Callable[[str, int, bytes], None]


def get_benchmark(language: str, word: str) -> Benchmark | None:
    """Load a benchmark using Redis as a disposable cache."""
    normalized_word = word.casefold()
    key = f"benchmark:{language.casefold()}:{normalized_word}"
    cached = client.get(key)
    if cached is not None:
        value = json.loads(cached)
        return Benchmark(**value)

    benchmark = find_benchmark(language, normalized_word)
    if benchmark is None:
        return None

    client.setex(
        key,
        3_600,
        json.dumps(asdict(benchmark), ensure_ascii=False),
    )
    return benchmark


def ensure_group(stream: str, group: str) -> None:
    try:
        client.xgroup_create(stream, group, id="0", mkstream=True)
    except ResponseError as error:
        if "BUSYGROUP" not in str(error):
            raise


def consume_frames(
    session_id: str,
    process_frame: FrameProcessor,
    *,
    stop_after_idle: bool = False,
) -> None:
    """Consume and acknowledge frames only after processing succeeds."""
    stream = f"audio:{session_id}"
    group = "pronunciation-workers"
    consumer = f"{socket.gethostname()}-{session_id[:8]}"
    ensure_group(stream, group)

    while True:
        batches = client.xreadgroup(
            groupname=group,
            consumername=consumer,
            streams={stream: ">"},
            count=25,
            block=1_000,
        )
        if not batches:
            if stop_after_idle:
                return
            continue

        for _, messages in batches:
            for message_id, fields in messages:
                sequence = int(fields[b"sequence"])
                pcm = fields[b"pcm_s16le"]
                process_frame(session_id, sequence, pcm)
                client.xack(stream, group, message_id)


def persist_final_result(
    session_id: str,
    benchmark: Benchmark,
    result: PronunciationResult,
) -> int:
    """Persist one final result and publish its versioned UI event."""
    attempt = save_attempt(session_id, benchmark.id, result)
    event: dict[str, Any] = {
        "type": "pronunciation.evaluated",
        "version": 1,
        "attempt_id": attempt.id,
        "phase": "evaluated",
        "expected_word": benchmark.word,
        "expected_phonemes": benchmark.expected_phonemes,
        "detected_phonemes": result.detected_phonemes,
        "confidence": result.overall_confidence,
        "score": result.score,
        "benchmark_version": benchmark.version,
        "engine_version": result.engine_version,
    }
    update_session_state(session_id, event)
    publish_ui_event(session_id, event)
    return attempt.id


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Consume frames for a session (requires an engine/VAD handler)."
    )
    parser.add_argument("session_id")
    args = parser.parse_args()
    raise SystemExit(
        f"Session {args.session_id}: configure a VAD/engine FrameProcessor "
        "and call consume_frames()."
    )


if __name__ == "__main__":
    _main()
