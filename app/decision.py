"""Deterministic policy for turning acoustic evidence into a decision."""

from dataclasses import dataclass
from enum import StrEnum
from math import isfinite


DECISION_VERSION = "acoustic-thresholds-v1"


class DecisionStatus(StrEnum):
    CORRECT = "CORRECT"
    INCORRECT = "INCORRECT"
    UNDECIDABLE = "UNDECIDABLE"
    RETRY = "RETRY"


class ReasonCode(StrEnum):
    WITHIN_BENCHMARK = "WITHIN_BENCHMARK"
    OUTSIDE_BENCHMARK = "OUTSIDE_BENCHMARK"
    BORDERLINE_DISTANCE = "BORDERLINE_DISTANCE"
    CLEANING_RETRY_RECOMMENDED = "CLEANING_RETRY_RECOMMENDED"
    INVALID_EVIDENCE = "INVALID_EVIDENCE"


@dataclass(frozen=True)
class DecisionResult:
    status: DecisionStatus
    distance: float | None
    accept_threshold: float
    reject_threshold: float
    reason_codes: tuple[ReasonCode, ...]
    decision_version: str = DECISION_VERSION

    @property
    def correct(self) -> bool | None:
        if self.status is DecisionStatus.CORRECT:
            return True
        if self.status is DecisionStatus.INCORRECT:
            return False
        return None


def decide(
    distance: float | None,
    *,
    accept_threshold: float,
    reject_threshold: float,
    cleaning_status: str = "accept",
    decision_version: str = DECISION_VERSION,
) -> DecisionResult:
    """Classify a distance, giving unusable cleaning results precedence.

    Distances at the accept boundary are accepted and distances at the reject
    boundary are rejected. The open interval between them is deliberately
    undecidable.
    """

    if not (
        isfinite(accept_threshold)
        and isfinite(reject_threshold)
        and accept_threshold < reject_threshold
    ):
        raise ValueError(
            "accept_threshold and reject_threshold must be finite and ordered"
        )

    if cleaning_status.casefold() == "retry_recommended":
        return DecisionResult(
            status=DecisionStatus.RETRY,
            distance=distance,
            accept_threshold=accept_threshold,
            reject_threshold=reject_threshold,
            reason_codes=(ReasonCode.CLEANING_RETRY_RECOMMENDED,),
            decision_version=decision_version,
        )

    if distance is None or not isfinite(distance) or distance < 0:
        return DecisionResult(
            status=DecisionStatus.UNDECIDABLE,
            distance=distance,
            accept_threshold=accept_threshold,
            reject_threshold=reject_threshold,
            reason_codes=(ReasonCode.INVALID_EVIDENCE,),
            decision_version=decision_version,
        )
    if distance <= accept_threshold:
        status = DecisionStatus.CORRECT
        reasons = (ReasonCode.WITHIN_BENCHMARK,)
    elif distance >= reject_threshold:
        status = DecisionStatus.INCORRECT
        reasons = (ReasonCode.OUTSIDE_BENCHMARK,)
    else:
        status = DecisionStatus.UNDECIDABLE
        reasons = (ReasonCode.BORDERLINE_DISTANCE,)

    return DecisionResult(
        status=status,
        distance=distance,
        accept_threshold=accept_threshold,
        reject_threshold=reject_threshold,
        reason_codes=reasons,
        decision_version=decision_version,
    )
