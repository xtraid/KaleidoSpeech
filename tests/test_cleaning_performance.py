"""Small end-to-end performance check of the public cleaning boundary."""

from __future__ import annotations

from time import perf_counter

import numpy as np

from cleaning_contract import clean_redis_record


def test_cleaning_is_faster_than_real_time(
    benchmark_case: dict[str, object],
) -> None:
    iterations = 10
    elapsed: list[float] = []
    redis_record = benchmark_case["redis_record"]
    assert isinstance(redis_record, dict)

    for _ in range(iterations):
        started = perf_counter()
        clean_redis_record(redis_record)
        elapsed.append(perf_counter() - started)

    mean_seconds = float(np.mean(elapsed))
    p95_seconds = float(np.percentile(elapsed, 95))
    input_duration_seconds = len(redis_record[b"audio"]) / np.dtype(np.float32).itemsize / 16_000

    assert mean_seconds / input_duration_seconds < 1.0, (
        f"cleaning is slower than real time: mean={mean_seconds:.3f}s, "
        f"p95={p95_seconds:.3f}s"
    )
