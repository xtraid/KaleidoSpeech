"""Import ZIP word recordings and build provisional English benchmarks."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from app.acoustic_benchmark import build_benchmarks, import_archives
from app.temporal_benchmark import build_temporal_benchmarks
from scripts.init_db import main as initialize_database


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build speaker-disjoint acoustic word benchmarks."
    )
    parser.add_argument("archives", nargs="+", type=Path)
    parser.add_argument("--locale", default="en-US")
    parser.add_argument("--limit-per-word", type=int)
    parser.add_argument("--version", default="1.0.0")
    parser.add_argument(
        "--skip-cleaning",
        action="store_true",
        help="Use only for quick pipeline experiments.",
    )
    arguments = parser.parse_args()

    initialize_database()
    summary = import_archives(
        arguments.archives,
        locale=arguments.locale,
        limit_per_word=arguments.limit_per_word,
        apply_cleaning=not arguments.skip_cleaning,
    )
    benchmarks = build_benchmarks(
        locale=arguments.locale,
        version=arguments.version,
    )
    temporal_benchmarks = build_temporal_benchmarks(
        locale=arguments.locale,
        version=f"dtw-{arguments.version}",
    )
    print(
        json.dumps(
            {
                "import": asdict(summary),
                "benchmarks": [
                    {
                        "word": benchmark.word,
                        "samples": benchmark.sample_count,
                        "speakers": benchmark.speaker_count,
                        "accept_threshold": benchmark.accept_threshold,
                        "reject_threshold": benchmark.reject_threshold,
                        "version": benchmark.version,
                    }
                    for benchmark in benchmarks
                ],
                "temporal_benchmarks": [
                    {
                        "word": benchmark.word,
                        "samples": benchmark.sample_count,
                        "speakers": benchmark.speaker_count,
                        "prototypes": len(benchmark.prototypes),
                        "accept_threshold": benchmark.accept_threshold,
                        "reject_threshold": benchmark.reject_threshold,
                        "margin_threshold": benchmark.margin_threshold,
                        "version": benchmark.version,
                    }
                    for benchmark in temporal_benchmarks
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
