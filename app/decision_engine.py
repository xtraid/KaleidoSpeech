"""Unified, ML-ready pronunciation decision interface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from app.phonetic_pipeline import (
    PronunciationDecision as DecisionResult,
    QualityEvidence as QualityResult,
    ScoredPhone as PhonemeScore,
    WordIdentityEvidence as WordIdentityResult,
    decide_pronunciation,
)


@dataclass(frozen=True)
class DecisionSignals:
    target_word: str
    quality: QualityResult
    word_identity: WordIdentityResult | None
    phoneme_scores: tuple[PhonemeScore, ...] = ()
    dtw_distance: float | None = None
    versions: dict[str, str] | None = None


class DecisionEngine(Protocol):
    def decide(self, signals: DecisionSignals) -> DecisionResult: ...


class RuleBasedDecisionEngine:
    """Conservative baseline: positive phonetic evidence is mandatory."""

    def decide(self, signals: DecisionSignals) -> DecisionResult:
        return decide_pronunciation(
            target_word=signals.target_word,
            quality=signals.quality,
            identity=signals.word_identity,
            phones=signals.phoneme_scores,
            versions=signals.versions,
        )


__all__ = [
    "DecisionEngine",
    "DecisionResult",
    "DecisionSignals",
    "RuleBasedDecisionEngine",
]
