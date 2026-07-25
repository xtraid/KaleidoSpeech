"""Stable feature extraction for future supervised decision models."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from statistics import mean
from typing import Any


FEATURE_SCHEMA_VERSION = "decision-signals-v1"
FEATURE_NAMES = (
    "word_identity_confidence",
    "pronunciation_score",
    "phoneme_score_mean",
    "phoneme_score_min",
    "alignment_confidence_mean",
    "alignment_confidence_min",
    "duration_ms",
    "voice_frame_ratio",
    "snr_db",
)


@dataclass(frozen=True)
class DecisionFeatureVector:
    values: tuple[float, ...]
    schema_version: str = FEATURE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if len(self.values) != len(FEATURE_NAMES) or not all(
            isfinite(value) for value in self.values
        ):
            raise ValueError("Feature vector is incomplete or non-finite")


def extract_decision_features(signals: dict[str, Any]) -> DecisionFeatureVector:
    phones = signals.get("phones") or []
    phone_scores = [
        float(phone["score"])
        for phone in phones
        if isinstance(phone.get("score"), (int, float))
    ]
    phone_confidences = [
        float(phone["confidence"])
        for phone in phones
        if isinstance(phone.get("confidence"), (int, float))
    ]
    quality = signals.get("quality") or {}

    def value(name: str, default: float = 0.0) -> float:
        candidate = signals.get(name, default)
        return float(candidate) if isinstance(candidate, (int, float)) else default

    return DecisionFeatureVector(
        (
            value("word_identity_confidence"),
            value("pronunciation_score"),
            mean(phone_scores) if phone_scores else 0.0,
            min(phone_scores) if phone_scores else 0.0,
            mean(phone_confidences) if phone_confidences else 0.0,
            min(phone_confidences) if phone_confidences else 0.0,
            value("duration_ms"),
            value("voice_frame_ratio"),
            float(quality.get("snr_db") or 0.0),
        )
    )
