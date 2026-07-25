"""Calibration for the closed-vocabulary wrong-word margin."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class LabeledWordDistances:
    target_word: str
    spoken_word: str
    distances: dict[str, float]
    speaker_id: str


@dataclass(frozen=True)
class WordGateCalibration:
    mismatch_margin_threshold: float
    wrong_word_false_acceptance_rate: float
    correct_word_false_rejection_rate: float
    sample_count: int


def calibrate_mismatch_margin(
    samples: Iterable[LabeledWordDistances],
    *,
    maximum_wrong_word_false_acceptance_rate: float = 0.02,
) -> WordGateCalibration:
    """Choose the smallest winner margin that meets the wrong-word FAR budget."""
    observations: list[tuple[bool, bool, float]] = []
    for sample in samples:
        if not sample.speaker_id or sample.target_word not in sample.distances:
            raise ValueError("Every sample needs speaker and target distance")
        if len(sample.distances) < 2:
            raise ValueError("Closed-vocabulary calibration needs competitors")
        ordered = sorted(sample.distances.items(), key=lambda item: item[1])
        if not all(np.isfinite(distance) for _, distance in ordered):
            raise ValueError("Distances must be finite")
        winner = ordered[0][0]
        margin = ordered[1][1] - ordered[0][1]
        observations.append(
            (
                sample.spoken_word != sample.target_word,
                winner == sample.target_word,
                float(margin),
            )
        )
    if not observations or not any(item[0] for item in observations):
        raise ValueError("Calibration requires wrong-word examples")

    margins = sorted({margin for _, _, margin in observations})
    candidates = [0.0, *margins, float(np.nextafter(margins[-1], np.inf))]
    evaluated: list[tuple[float, float, float]] = []
    for threshold in candidates:
        wrong = [item for item in observations if item[0]]
        correct = [item for item in observations if not item[0]]
        false_accepts = sum(
            target_won and margin >= threshold
            for _, target_won, margin in wrong
        )
        false_rejects = sum(
            not target_won or margin < threshold
            for _, target_won, margin in correct
        )
        far = false_accepts / len(wrong)
        frr = false_rejects / len(correct) if correct else 0.0
        evaluated.append((far, frr, float(threshold)))
    feasible = [
        item
        for item in evaluated
        if item[0] <= maximum_wrong_word_false_acceptance_rate
    ]
    if not feasible:
        raise ValueError("No margin meets the wrong-word false acceptance budget")
    far, frr, threshold = min(feasible, key=lambda item: (item[1], item[2]))
    return WordGateCalibration(threshold, far, frr, len(observations))
