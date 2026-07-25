"""Speaker-disjoint threshold calibration utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class LabeledPhoneScore:
    phone: str
    score: float
    is_error: bool
    speaker_id: str


@dataclass(frozen=True)
class CalibratedPhoneThreshold:
    phone: str
    threshold: float
    false_acceptance_rate: float
    false_rejection_rate: float
    sample_count: int


def calibrate_phone_thresholds(
    samples: Iterable[LabeledPhoneScore],
    *,
    maximum_false_acceptance_rate: float = 0.10,
    maximum_false_rejection_rate: float = 0.05,
) -> dict[str, CalibratedPhoneThreshold]:
    """Grid-search fail/pass cutoff using validation data only.

    Scores at or below the cutoff are predicted as phoneme errors. Among
    feasible cutoffs, the one with the lowest total error is selected.
    """
    grouped: dict[str, list[LabeledPhoneScore]] = {}
    for sample in samples:
        if not np.isfinite(sample.score) or not sample.speaker_id:
            raise ValueError("Calibration samples need finite scores and speakers")
        grouped.setdefault(sample.phone, []).append(sample)

    calibrated: dict[str, CalibratedPhoneThreshold] = {}
    for phone, phone_samples in grouped.items():
        errors = [sample for sample in phone_samples if sample.is_error]
        correct = [sample for sample in phone_samples if not sample.is_error]
        if not errors or not correct:
            raise ValueError(f"Phone {phone!r} needs both error and correct samples")
        values = sorted({sample.score for sample in phone_samples})
        epsilon = np.finfo(float).eps
        candidates = [values[0] - epsilon, *values, values[-1] + epsilon]
        evaluated = []
        for threshold in candidates:
            false_accepts = sum(sample.score > threshold for sample in errors)
            false_rejects = sum(sample.score <= threshold for sample in correct)
            far = false_accepts / len(errors)
            frr = false_rejects / len(correct)
            evaluated.append((far, frr, threshold))
        feasible = [
            result
            for result in evaluated
            if result[0] <= maximum_false_acceptance_rate
            and result[1] <= maximum_false_rejection_rate
        ]
        if not feasible:
            raise ValueError(f"No threshold meets error budgets for {phone!r}")
        far, frr, threshold = min(
            feasible, key=lambda result: (result[0] + result[1], result[0])
        )
        calibrated[phone] = CalibratedPhoneThreshold(
            phone=phone,
            threshold=float(threshold),
            false_acceptance_rate=far,
            false_rejection_rate=frr,
            sample_count=len(phone_samples),
        )
    return calibrated
