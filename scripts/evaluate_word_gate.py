"""Evaluate closed-vocabulary identity without tuning on the frozen test set."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from app.phonetic_pipeline import DtwWordIdentityGate
from app.temporal_benchmark import _distance_to_prototypes
from app.temporal_repository import (
    list_temporal_benchmarks,
    list_temporal_recordings,
)


@dataclass(frozen=True)
class WordGateReport:
    split: str
    sample_count: int
    match_accuracy: float
    mismatch_count: int
    uncertain_count: int
    wrong_word_pair_count: int
    wrong_word_false_acceptance_rate: float
    latency_ms_p50: float
    latency_ms_p95: float
    latency_ms_max: float
    threshold: float
    downsample_factor: int


def evaluate(
    split: str, *, threshold: float = 0.05, downsample_factor: int = 2
) -> WordGateReport:
    if split not in {"validation", "test"}:
        raise ValueError("Only validation or frozen test evaluation is allowed")
    benchmarks = list_temporal_benchmarks()
    words = tuple(benchmark.word for benchmark in benchmarks)
    if len(words) < 2:
        raise ValueError("At least two active word benchmarks are required")
    if downsample_factor < 1:
        raise ValueError("downsample_factor must be positive")

    # The gate contract accepts bytes, while this evaluator deliberately uses
    # already-persisted temporal evidence to isolate identity latency.
    current_sequence: np.ndarray | None = None

    def stored_distances(_: bytes) -> dict[str, float]:
        if current_sequence is None:
            raise RuntimeError("No evaluation sequence")
        return {
            benchmark.word: _distance_to_prototypes(
                current_sequence[::downsample_factor],
                tuple(
                    prototype[::downsample_factor]
                    for prototype in benchmark.prototypes
                ),
            )
            for benchmark in benchmarks
        }

    gate = DtwWordIdentityGate(
        stored_distances, mismatch_margin_threshold=threshold
    )
    statuses: list[str] = []
    latencies: list[float] = []
    recordings = list_temporal_recordings(dataset_split=split)
    for recording in recordings:
        current_sequence = recording.sequence
        started = perf_counter()
        result = gate.evaluate(b"", recording.word)
        latencies.append((perf_counter() - started) * 1_000)
        statuses.append(result.status)

    if not statuses:
        raise ValueError(f"No {split} temporal recordings")
    mismatch_count = statuses.count("MISMATCH")
    pair_count = len(statuses) * (len(words) - 1)
    latency = np.asarray(latencies)
    return WordGateReport(
        split=split,
        sample_count=len(statuses),
        match_accuracy=statuses.count("MATCH") / len(statuses),
        mismatch_count=mismatch_count,
        uncertain_count=statuses.count("UNCERTAIN"),
        wrong_word_pair_count=pair_count,
        # A confident misclassification accepts exactly one wrong target.
        wrong_word_false_acceptance_rate=mismatch_count / pair_count,
        latency_ms_p50=float(np.percentile(latency, 50)),
        latency_ms_p95=float(np.percentile(latency, 95)),
        latency_ms_max=float(np.max(latency)),
        threshold=threshold,
        downsample_factor=downsample_factor,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--downsample-factor", type=int, default=2)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    report = asdict(
        evaluate(
            arguments.split,
            threshold=arguments.threshold,
            downsample_factor=arguments.downsample_factor,
        )
    )
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
