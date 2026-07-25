"""Create one annotated sentence benchmark from a reference WAV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import soundfile as sf

from app.sentence_repository import store_sentence_benchmark
from app.sentence_streaming import SentenceUnit, extract_streaming_features


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sentence_id")
    parser.add_argument("text")
    parser.add_argument("wav", type=Path)
    parser.add_argument(
        "units_json",
        type=Path,
        help='JSON array: [{"text":"hello","start_ms":0,"end_ms":400}]',
    )
    parser.add_argument("--language", default="en")
    parser.add_argument("--locale", default="en-US")
    parser.add_argument("--error-threshold", type=float, default=0.55)
    parser.add_argument("--final-threshold", type=float, default=0.40)
    parser.add_argument("--version", default="1")
    args = parser.parse_args()

    samples, sample_rate = sf.read(args.wav, dtype="int16")
    if sample_rate != 16_000 or samples.ndim != 1:
        parser.error("reference WAV must be mono 16 kHz")
    pcm = samples.astype("<i2").tobytes()
    chunk_bytes = 16_000 * 200 // 1_000 * 2
    pcm = pcm[: len(pcm) // chunk_bytes * chunk_bytes]
    if not pcm:
        parser.error("reference WAV must contain at least 200 ms")
    units = tuple(
        SentenceUnit(**item)
        for item in json.loads(args.units_json.read_text(encoding="utf-8"))
    )
    benchmark_id = store_sentence_benchmark(
        args.sentence_id,
        args.text,
        args.language,
        args.locale,
        extract_streaming_features(pcm),
        units,
        error_threshold=args.error_threshold,
        final_threshold=args.final_threshold,
        version=args.version,
    )
    print(json.dumps({
        "benchmark_id": benchmark_id,
        "sentence_id": args.sentence_id,
        "units": len(units),
    }))


if __name__ == "__main__":
    main()
