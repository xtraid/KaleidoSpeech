"""Final-decision fusion for phonetic, DTW and single-word scorers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any, Callable

import numpy as np

from app.config import get_settings
from app.model_registry import resolve_active_model


DEFAULT_WEIGHTS = {"phonetic": 0.60, "dtw": 0.25, "single_word": 0.15}
SUPPORTED_STATUSES = {"CORRECT", "INCORRECT", "UNDECIDABLE", "RETRY"}


@dataclass(frozen=True)
class ComponentResult:
    status: str
    score: float
    confidence: float
    available: bool = True
    reason_codes: tuple[str, ...] = ()


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _vote(status: str, confidence: float | None) -> float:
    """Map a categorical verdict to calibrated, neutral-centred evidence."""
    strength = _clamp(confidence if confidence is not None else 1.0)
    if status == "CORRECT":
        return 0.5 + 0.5 * strength
    if status == "INCORRECT":
        return 0.5 - 0.5 * strength
    return 0.5


def _phonetic_component(result: dict[str, Any]) -> ComponentResult:
    status = str(result.get("status", "UNDECIDABLE"))
    phones = result.get("phones") or []
    confidences = [
        float(phone["confidence"])
        for phone in phones
        if isinstance(phone.get("confidence"), (int, float))
    ]
    confidence = (
        sum(confidences) / len(confidences)
        if confidences
        else float(result.get("word_identity_confidence") or 0.0)
    )
    return ComponentResult(
        status=status,
        score=_vote(status, confidence),
        confidence=_clamp(confidence),
        reason_codes=tuple(str(code) for code in result.get("reason_codes", ())),
    )


def _default_dtw_scorer(word: str, pcm: bytes) -> ComponentResult:
    from app.temporal_benchmark import evaluate_temporal_word_pcm

    evaluation = evaluate_temporal_word_pcm(word, pcm)
    decision = evaluation.decision
    margin_threshold = max(float(evaluation.benchmark.margin_threshold), 1e-9)
    confidence = (
        min(1.0, abs(float(decision.margin)) / margin_threshold)
        if decision.margin is not None
        else 0.0
    )
    status = str(decision.status)
    return ComponentResult(
        status=status,
        score=_vote(status, confidence),
        confidence=confidence,
        reason_codes=tuple(decision.reason_codes),
    )


def _default_single_word_scorer(word: str, pcm: bytes) -> ComponentResult:
    from app.acoustic_benchmark import evaluate_word_pcm

    evaluation = evaluate_word_pcm(word, pcm)
    status = str(evaluation.decision.status)
    confidence = _clamp(evaluation.confidence or 0.0)
    return ComponentResult(
        status=status,
        score=_vote(status, confidence),
        confidence=confidence,
        reason_codes=tuple(
            str(reason.value) for reason in evaluation.decision.reason_codes
        ),
    )


class EnsembleDecisionEngine:
    def __init__(
        self,
        model_config: dict[str, Any],
        *,
        dtw_scorer: Callable[[str, bytes], ComponentResult] = _default_dtw_scorer,
        single_word_scorer: Callable[
            [str, bytes], ComponentResult
        ] = _default_single_word_scorer,
    ) -> None:
        self.config = model_config
        self.version = str(model_config["version"])
        self.kind = str(model_config.get("kind", "rule_based"))
        configured_weights = model_config.get("ensemble_weights") or DEFAULT_WEIGHTS
        self.weights = {
            name: float(configured_weights.get(name, DEFAULT_WEIGHTS[name]))
            for name in DEFAULT_WEIGHTS
        }
        if any(weight < 0 for weight in self.weights.values()):
            raise ValueError("Ensemble weights cannot be negative")
        if sum(self.weights.values()) <= 0:
            raise ValueError("At least one ensemble weight must be positive")
        self.activation_floor = float(model_config.get("activation_floor", 0.65))
        if not 0 <= self.activation_floor <= 1:
            raise ValueError("Ensemble activation_floor must be between zero and one")
        self.dtw_scorer = dtw_scorer
        self.single_word_scorer = single_word_scorer

    def evaluate(
        self,
        phonetic_result: dict[str, Any],
        pcm: bytes,
        *,
        voice_frame_ratio: float,
    ) -> dict[str, Any]:
        if self.kind == "rule_based":
            return phonetic_result

        target_word = str(phonetic_result.get("target_word") or "")
        components = {"phonetic": _phonetic_component(phonetic_result)}
        for name, scorer in (
            ("dtw", self.dtw_scorer),
            ("single_word", self.single_word_scorer),
        ):
            try:
                components[name] = scorer(target_word, pcm)
            except (LookupError, ValueError):
                components[name] = ComponentResult(
                    status="UNDECIDABLE",
                    score=0.5,
                    confidence=0.0,
                    available=False,
                    reason_codes=("MODEL_UNAVAILABLE",),
                )

        available_weight = sum(
            self.weights[name]
            for name, component in components.items()
            if component.available
        )
        weighted_score = (
            sum(
                self.weights[name] * component.score
                for name, component in components.items()
                if component.available
            )
            / available_weight
        )
        final_status = (
            "CORRECT" if weighted_score >= self.activation_floor else "INCORRECT"
        )
        decision_confidence = abs(weighted_score - 0.5) * 2.0

        # Audio-quality failures are not safe to override with an ensemble.
        if phonetic_result.get("status") == "RETRY":
            final_status = "RETRY"
        if final_status not in SUPPORTED_STATUSES:
            final_status = "UNDECIDABLE"

        result = dict(phonetic_result)
        result["status"] = final_status
        result["decision_confidence"] = round(_clamp(decision_confidence), 6)
        result["ensemble"] = {
            "source": "weighted_ensemble",
            "weighted_score": round(weighted_score, 6),
            "activation_floor": self.activation_floor,
            "activated_score": int(weighted_score >= self.activation_floor),
            "duration_ms": round(len(pcm) / 32.0, 3),
            "voice_frame_ratio": round(_clamp(voice_frame_ratio), 6),
            "weights": self.weights,
            "components": {
                name: asdict(component) for name, component in components.items()
            },
        }
        existing_reasons = [
            str(reason) for reason in phonetic_result.get("reason_codes", ())
        ]
        result["reason_codes"] = [
            *existing_reasons,
            "WEIGHTED_ENSEMBLE_DECISION",
        ]
        versions = dict(phonetic_result.get("versions") or {})
        versions["phonetic_decision"] = versions.get("decision", "unknown")
        versions["decision"] = self.version
        result["versions"] = versions
        self._update_target_word(result, target_word, final_status, weighted_score)
        return result

    @staticmethod
    def _update_target_word(
        result: dict[str, Any], target_word: str, status: str, score: float
    ) -> None:
        words = result.get("words")
        if not isinstance(words, list):
            return
        for word_result in words:
            if str(word_result.get("word", "")).casefold() == target_word.casefold():
                word_result["status"] = status
                word_result["ensemble_score"] = round(score, 6)
                break

@lru_cache(maxsize=1)
def get_ensemble_decision_engine() -> EnsembleDecisionEngine:
    settings = get_settings()
    config = resolve_active_model(settings.decision_registry_path)
    return EnsembleDecisionEngine(config)


def voice_frame_ratio(pcm: bytes, *, frame_samples: int = 640) -> float:
    samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
    if samples.size == 0:
        return 0.0
    threshold = get_settings().voice_rms_threshold
    voiced = 0
    frames = 0
    for start in range(0, samples.size, frame_samples):
        frame = samples[start:start + frame_samples]
        if frame.size == 0:
            continue
        frames += 1
        if float(np.sqrt(np.mean(frame * frame))) >= threshold:
            voiced += 1
    return voiced / frames if frames else 0.0


__all__ = [
    "ComponentResult",
    "EnsembleDecisionEngine",
    "get_ensemble_decision_engine",
    "voice_frame_ratio",
]
