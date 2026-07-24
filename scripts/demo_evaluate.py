"""Evaluate one WAV against a previously built acoustic word benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import soundfile as sf

from app.temporal_benchmark import evaluate_temporal_word_pcm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("word")
    parser.add_argument("wav", type=Path)
    parser.add_argument("--locale", default="en-US")
    parser.add_argument("--session-id", default="local-demo")
    arguments = parser.parse_args()

    samples, sample_rate = sf.read(arguments.wav, dtype="int16")
    if sample_rate != 16_000 or samples.ndim != 1:
        raise SystemExit("The demo requires mono 16 kHz PCM WAV input")
    evaluation = evaluate_temporal_word_pcm(
        arguments.word,
        samples.astype("<i2").tobytes(),
        locale=arguments.locale,
        session_id=arguments.session_id,
    )
    print(
        json.dumps(
            {
                "word": evaluation.benchmark.word,
                "status": evaluation.decision.status.value,
                "distance": evaluation.decision.target_distance,
                "competitor_word": evaluation.decision.competitor_word,
                "competitor_distance": evaluation.decision.competitor_distance,
                "margin": evaluation.decision.margin,
                "distances": evaluation.distances,
                "score": evaluation.score,
                "reasons": list(evaluation.decision.reason_codes),
                "benchmark_version": evaluation.benchmark.version,
                "decision_version": evaluation.decision.decision_version,
                "attempt_id": evaluation.attempt_id,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
