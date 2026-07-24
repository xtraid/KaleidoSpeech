"""Redis consumer primitives and final pronunciation-result handling."""

import argparse
import json
import socket
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import soundfile as sf
from redis.exceptions import ResponseError

from app.acoustic_benchmark import evaluate_word_pcm
from app.temporal_benchmark import evaluate_temporal_word_pcm
from app.benchmark_repository import (
    Benchmark,
    find_benchmark,
    save_attempt,
)
from app.pronunciation_engine import PronunciationResult
from app.redis_bus import (
    client,
    get_json,
    publish_ui_event,
    set_json,
    update_session_state,
)
from app.config import get_settings


FrameProcessor = Callable[[str, int, bytes], None]


def parse_audio_frame(fields: dict[bytes, bytes]) -> tuple[int, bytes]:
    """Validate and decode exactly one record from the Redis audio stream."""
    settings = get_settings()
    required = {
        b"sequence",
        b"captured_at_ns",
        b"sample_rate",
        b"frame_ms",
        b"pcm_s16le",
    }
    missing = required.difference(fields)
    if missing:
        names = ", ".join(sorted(field.decode() for field in missing))
        raise ValueError(f"Redis audio frame is missing: {names}")

    if int(fields[b"sample_rate"]) != settings.sample_rate:
        raise ValueError("Redis audio frame has an unexpected sample rate")
    if int(fields[b"frame_ms"]) != settings.frame_ms:
        raise ValueError("Redis audio frame has an unexpected duration")

    pcm = fields[b"pcm_s16le"]
    if len(pcm) != settings.bytes_per_frame:
        raise ValueError(
            f"Expected {settings.bytes_per_frame} PCM bytes, got {len(pcm)}"
        )
    return int(fields[b"sequence"]), pcm


def assemble_word_window(records: list[dict[bytes, bytes]]) -> bytes:
    """Concatenate an ordered, contiguous Redis frame sequence."""
    if not records:
        raise ValueError("A word window must contain at least one frame")

    decoded = [parse_audio_frame(fields) for fields in records]
    sequences = [sequence for sequence, _ in decoded]
    expected = list(range(sequences[0], sequences[0] + len(sequences)))
    if sequences != expected:
        raise ValueError("Redis audio frame sequence is not contiguous")
    return b"".join(pcm for _, pcm in decoded)


def get_benchmark(language: str, word: str) -> Benchmark | None:
    """Load a benchmark using Redis as a disposable cache."""
    normalized_word = word.casefold()
    key = f"benchmark:{language.casefold()}:{normalized_word}"
    cached = get_json(key, redis_client=client)
    if cached is not None:
        return Benchmark(**cached)

    benchmark = find_benchmark(language, normalized_word)
    if benchmark is None:
        return None

    set_json(
        key,
        asdict(benchmark),
        ttl_seconds=3_600,
        redis_client=client,
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

    claimed = client.xautoclaim(
        stream,
        group,
        consumer,
        min_idle_time=30_000,
        start_id="0-0",
        count=25,
    )
    pending_messages = claimed[1] if len(claimed) > 1 else []
    for message_id, fields in pending_messages:
        sequence, pcm = parse_audio_frame(fields)
        process_frame(session_id, sequence, pcm)
        client.xack(stream, group, message_id)

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
                sequence, pcm = parse_audio_frame(fields)
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
        description="Pronunciation worker and local acoustic evaluation."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    evaluate_parser = subparsers.add_parser(
        "evaluate-wav",
        help="Run the complete local benchmark decision path for one WAV.",
    )
    evaluate_parser.add_argument("word")
    evaluate_parser.add_argument("wav", type=Path)
    evaluate_parser.add_argument("--locale", default="en-US")
    evaluate_parser.add_argument("--session-id", default="worker-local")
    evaluate_parser.add_argument(
        "--engine",
        choices=("temporal", "summary"),
        default="temporal",
    )
    args = parser.parse_args()

    samples, sample_rate = sf.read(args.wav, dtype="int16")
    if sample_rate != get_settings().sample_rate or samples.ndim != 1:
        raise SystemExit("Expected a mono 16 kHz WAV")
    pcm = samples.astype("<i2").tobytes()
    if args.engine == "temporal":
        temporal = evaluate_temporal_word_pcm(
            args.word,
            pcm,
            locale=args.locale,
            session_id=args.session_id,
        )
        payload = {
            "engine": "temporal-dtw",
            "word": temporal.benchmark.word,
            "status": temporal.decision.status.value,
            "distance": temporal.decision.target_distance,
            "competitor_word": temporal.decision.competitor_word,
            "competitor_distance": temporal.decision.competitor_distance,
            "margin": temporal.decision.margin,
            "score": temporal.score,
            "reason_codes": list(temporal.decision.reason_codes),
            "distances": temporal.distances,
            "attempt_id": temporal.attempt_id,
        }
    else:
        summary = evaluate_word_pcm(
            args.word,
            pcm,
            locale=args.locale,
            session_id=args.session_id,
        )
        payload = {
            "engine": "summary-mfcc",
            "word": summary.benchmark.word,
            "status": summary.decision.status.value,
            "distance": summary.decision.distance,
            "score": summary.score,
            "confidence": summary.confidence,
            "reason_codes": [
                reason.value for reason in summary.decision.reason_codes
            ],
            "attempt_id": summary.attempt_id,
        }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    _main()
