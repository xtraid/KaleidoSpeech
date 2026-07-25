"""Guarded Random Forest training; requires reviewed speaker-disjoint data."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Literal

import numpy as np

from app.ml_features import FEATURE_NAMES, FEATURE_SCHEMA_VERSION


Split = Literal["train", "validation", "test"]


@dataclass(frozen=True)
class TrainingExample:
    attempt_id: str
    speaker_hash: str
    split: Split
    features: tuple[float, ...]
    label: str
    rule_based_label: str


@dataclass(frozen=True)
class TrainingReport:
    model_version: str
    feature_schema_version: str
    train_count: int
    validation_count: int
    test_count: int
    validation_accuracy: float
    test_accuracy: float
    rule_based_test_accuracy: float
    artifact_sha256: str


def validate_training_examples(
    examples: list[TrainingExample], *, minimum_examples: int = 500
) -> None:
    if len(examples) < minimum_examples:
        raise ValueError(f"At least {minimum_examples} reviewed examples are required")
    speakers: dict[str, str] = {}
    split_counts = {"train": 0, "validation": 0, "test": 0}
    for example in examples:
        if len(example.features) != len(FEATURE_NAMES):
            raise ValueError("Feature vector does not match the versioned schema")
        if not np.isfinite(example.features).all():
            raise ValueError("Training features must be finite")
        previous = speakers.setdefault(example.speaker_hash, example.split)
        if previous != example.split:
            raise ValueError("A speaker appears in multiple dataset splits")
        split_counts[example.split] += 1
    if not all(split_counts.values()):
        raise ValueError("Train, validation and frozen test splits are required")


def train_random_forest(
    examples: list[TrainingExample],
    *,
    artifact_path: Path,
    model_version: str,
) -> TrainingReport:
    """Fit on train only, inspect validation, and report frozen-test metrics."""
    validate_training_examples(examples)
    try:
        import joblib
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score
    except ImportError as error:
        raise RuntimeError("Install with: uv sync --extra ml") from error

    subsets = {
        split: [example for example in examples if example.split == split]
        for split in ("train", "validation", "test")
    }
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(
        np.asarray([item.features for item in subsets["train"]]),
        [item.label for item in subsets["train"]],
    )

    def accuracy(split: str) -> float:
        items = subsets[split]
        predicted = model.predict(np.asarray([item.features for item in items]))
        return float(accuracy_score([item.label for item in items], predicted))

    test_items = subsets["test"]
    rule_accuracy = float(
        np.mean([item.label == item.rule_based_label for item in test_items])
    )
    bundle = {
        "model": model,
        "model_version": model_version,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_names": FEATURE_NAMES,
    }
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, artifact_path)
    digest = sha256(artifact_path.read_bytes()).hexdigest()
    report = TrainingReport(
        model_version=model_version,
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        train_count=len(subsets["train"]),
        validation_count=len(subsets["validation"]),
        test_count=len(subsets["test"]),
        validation_accuracy=accuracy("validation"),
        test_accuracy=accuracy("test"),
        rule_based_test_accuracy=rule_accuracy,
        artifact_sha256=digest,
    )
    artifact_path.with_suffix(artifact_path.suffix + ".report.json").write_text(
        json.dumps(asdict(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report
