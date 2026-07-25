"""Redis consumer primitives and final pronunciation-result handling."""

import argparse
import json
import logging
from pathlib import Path

import soundfile as sf

from app.acoustic_benchmark import evaluate_word_pcm
from app.config import get_settings
from app.temporal_benchmark import evaluate_temporal_word_pcm


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
    logging.getLogger(__name__).info(
        "worker_stopped payload=%s", json.dumps(payload, sort_keys=True)
    )


if __name__ == "__main__":
    _main()
