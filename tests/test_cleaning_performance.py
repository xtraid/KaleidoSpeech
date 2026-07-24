"""Small end-to-end performance check of the public cleaning boundary."""

from __future__ import annotations

from time import perf_counter

import numpy as np

from cleaning_contract import clean_redis_records


def test_cleaning_is_faster_than_real_time(
    benchmark_case: dict[str, object],
) -> None:
    iterations = 10
    elapsed: list[float] = []
    redis_records = benchmark_case["redis_records"]
    input_pcm = benchmark_case["input_pcm_s16le"]
    assert isinstance(redis_records, list)
    assert isinstance(input_pcm, bytes)

    for _ in range(iterations):
        started = perf_counter()
        clean_redis_records(redis_records)
        elapsed.append(perf_counter() - started)

    mean_seconds = float(np.mean(elapsed))
    p95_seconds = float(np.percentile(elapsed, 95))
    input_duration_seconds = len(input_pcm) / np.dtype("<i2").itemsize / 16_000

    assert mean_seconds / input_duration_seconds < 1.0, (
        f"cleaning is slower than real time: mean={mean_seconds:.3f}s, "
        f"p95={p95_seconds:.3f}s"
    )
