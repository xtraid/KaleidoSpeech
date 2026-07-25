"""Train a candidate decision forest from a reviewed JSONL export."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from app.ml_training import TrainingExample, train_random_forest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--version", required=True)
    arguments = parser.parse_args()
    examples = [
        TrainingExample(**json.loads(line))
        for line in arguments.dataset.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    report = train_random_forest(
        examples,
        artifact_path=arguments.artifact,
        model_version=arguments.version,
    )
    print(json.dumps(asdict(report), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
