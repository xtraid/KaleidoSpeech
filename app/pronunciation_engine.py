"""Contract for a real phonetic pronunciation engine."""

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class PronunciationResult:
    detected_phonemes: list[str]
    phoneme_confidences: list[float]
    overall_confidence: float
    score: float | None
    engine_version: str


class PronunciationEngine(Protocol):
    def evaluate(
        self,
        pcm_s16le: bytes,
        sample_rate: int,
        expected_phonemes: list[str],
        accepted_variants: list[list[str]],
    ) -> PronunciationResult:
        """Evaluate one complete word window, never an individual frame."""
        ...
