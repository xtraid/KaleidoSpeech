"""Measure pinned phoneme-model load, inference latency and process RSS."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import resource
from statistics import median
from time import perf_counter

from app.pronunciation_engine import TransformersWav2Vec2PhonemeModel


@dataclass(frozen=True)
class Profile:
    model_version: str
    load_seconds: float
    inference_ms_p50: float
    inference_ms_p95: float
    peak_rss_mib: float
    runs: int


def profile(
    runs: int, *, device: str = "cpu", quantize_cpu: bool = False
) -> Profile:
    if runs < 2:
        raise ValueError("At least two inference runs are required")
    started = perf_counter()
    model = TransformersWav2Vec2PhonemeModel(
        device=device,
        quantize_cpu=quantize_cpu,
    )
    load_seconds = perf_counter() - started
    pcm = bytes(16_000 * 2)
    latencies = []
    for _ in range(runs):
        started = perf_counter()
        model.infer(pcm, 16_000)
        latencies.append((perf_counter() - started) * 1_000)
    ordered = sorted(latencies)
    p95_index = min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))
    return Profile(
        model_version=model.version,
        load_seconds=load_seconds,
        inference_ms_p50=median(ordered),
        inference_ms_p95=ordered[p95_index],
        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        runs=runs,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--quantize-cpu",
        action="store_true",
        help="Quantize transformer Linear layers to dynamic INT8 on CPU",
    )
    arguments = parser.parse_args()
    print(
        json.dumps(
            asdict(
                profile(
                    arguments.runs,
                    device=arguments.device,
                    quantize_cpu=arguments.quantize_cpu,
                )
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
