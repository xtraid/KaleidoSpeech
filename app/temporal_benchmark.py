"""DTW benchmark construction and contrastive word decision."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import exp
from typing import Sequence

import numpy as np

from app.cleaning import clean_pcm_with_report
from app.decision import DecisionStatus
from app.temporal_features import (
    TEMPORAL_DISTANCE_VERSION,
    TEMPORAL_EXTRACTOR_VERSION,
    dtw_distance,
    extract_temporal_features,
)
from app.temporal_repository import (
    TemporalBenchmark,
    TemporalRecording,
    find_temporal_benchmark,
    list_temporal_benchmarks,
    list_temporal_recordings,
    save_temporal_attempt,
    store_temporal_benchmark,
)


TEMPORAL_DECISION_VERSION = "contrastive-dtw-v1"
DEFAULT_TASK = "word_pronunciation"


@dataclass(frozen=True)
class ContrastiveDecision:
    status: DecisionStatus
    target_word: str
    target_distance: float | None
    competitor_word: str | None
    competitor_distance: float | None
    margin: float | None
    reason_codes: tuple[str, ...]
    decision_version: str = TEMPORAL_DECISION_VERSION


@dataclass(frozen=True)
class TemporalEvaluation:
    benchmark: TemporalBenchmark
    decision: ContrastiveDecision
    distances: dict[str, float]
    score: float | None
    attempt_id: int | None


def _distance_to_prototypes(
    sequence: np.ndarray,
    prototypes: Sequence[np.ndarray],
) -> float:
    """Return the nearest DTW distance to a prototype collection."""
    return min(dtw_distance(sequence, prototype) for prototype in prototypes)


def _subsequence_distance_to_prototypes(
    sequence: np.ndarray,
    prototypes: Sequence[np.ndarray],
    *,
    length_ratios: tuple[float, ...] = (0.75, 1.0, 1.25),
) -> float:
    """Find a word prototype inside a longer utterance.

    This is the lightweight sentence-mode baseline: each prototype is compared
    with overlapping windows near its expected duration instead of stretching
    the complete sentence onto one isolated-word recording.
    """
    observed = np.asarray(sequence, dtype=np.float32)
    if observed.ndim != 2 or not len(observed):
        raise ValueError("sequence must be a non-empty feature matrix")
    best = float("inf")
    for prototype in prototypes:
        expected = np.asarray(prototype, dtype=np.float32)
        for ratio in length_ratios:
            window_length = max(3, round(len(expected) * ratio))
            if window_length >= len(observed):
                candidates = (observed,)
            else:
                hop = max(1, window_length // 5)
                starts = list(range(0, len(observed) - window_length + 1, hop))
                final_start = len(observed) - window_length
                if not starts or starts[-1] != final_start:
                    starts.append(final_start)
                candidates = (
                    observed[start : start + window_length] for start in starts
                )
            for candidate in candidates:
                best = min(best, dtw_distance(candidate, expected, band_ratio=1.0))
    return best


def _prototype_medoid_candidates(
    recordings: Sequence[TemporalRecording],
    *,
    max_candidates: int = 36,
    prototype_count: int = 4,
) -> list[TemporalRecording]:
    """Select actual-recording medoids with deterministic farthest-first spread."""
    if not recordings:
        raise ValueError("Cannot select prototypes from no recordings")
    ordered = sorted(recordings, key=lambda item: item.recording_id)
    if len(ordered) > max_candidates:
        indexes = np.linspace(0, len(ordered) - 1, max_candidates).astype(int)
        candidates = [ordered[int(index)] for index in indexes]
    else:
        candidates = ordered

    count = len(candidates)
    pairwise = np.zeros((count, count), dtype=np.float64)
    for left in range(count):
        for right in range(left + 1, count):
            distance = dtw_distance(
                candidates[left].sequence,
                candidates[right].sequence,
            )
            pairwise[left, right] = pairwise[right, left] = distance
    selected = [int(np.argmin(np.mean(pairwise, axis=1)))]
    while len(selected) < min(prototype_count, count):
        nearest = np.min(pairwise[:, selected], axis=1)
        nearest[selected] = -1
        selected.append(int(np.argmax(nearest)))
    # Refine farthest-first seeds into central real examples for each cluster.
    for _ in range(8):
        assignments = np.argmin(pairwise[:, selected], axis=1)
        refined: list[int] = []
        for cluster in range(len(selected)):
            members = np.flatnonzero(assignments == cluster)
            if not len(members):
                refined.append(selected[cluster])
                continue
            within = pairwise[np.ix_(members, members)]
            refined.append(int(members[int(np.argmin(np.mean(within, axis=1)))]))
        if refined == selected:
            break
        selected = refined
    return [candidates[index] for index in selected]


def _quantile(values: Sequence[float], probability: float) -> float:
    """Compute a calibration quantile while rejecting missing evidence."""
    if not values:
        raise ValueError("Cannot calibrate an empty distance distribution")
    return float(np.quantile(np.asarray(values), probability))


def build_temporal_benchmarks(
    *,
    locale: str = "en-US",
    task: str = DEFAULT_TASK,
    version: str = "2.0.0",
    prototype_count: int = 4,
) -> list[TemporalBenchmark]:
    """Build medoids, DTW thresholds, and contrastive margins from validation."""
    recordings = list_temporal_recordings(locale=locale)
    grouped: dict[str, list[TemporalRecording]] = defaultdict(list)
    for recording in recordings:
        grouped[recording.word].append(recording)
    if len(grouped) < 2:
        raise ValueError("At least two words are required for contrastive DTW")

    prototypes: dict[str, list[TemporalRecording]] = {}
    for word, items in grouped.items():
        training = [item for item in items if item.dataset_split == "train"]
        if len(training) < 2:
            raise ValueError(f"Word {word!r} has insufficient DTW training data")
        prototypes[word] = _prototype_medoid_candidates(
            training,
            prototype_count=prototype_count,
        )

    built: list[TemporalBenchmark] = []
    words = sorted(grouped)
    for word in words:
        training = [
            item for item in grouped[word] if item.dataset_split == "train"
        ]
        validation = [
            item for item in grouped[word] if item.dataset_split == "validation"
        ] or training[: min(20, len(training))]
        target_prototypes = [item.sequence for item in prototypes[word]]
        positive_distances: list[float] = []
        positive_margins: list[float] = []
        for recording in validation:
            target_distance = _distance_to_prototypes(
                recording.sequence,
                target_prototypes,
            )
            competitor_distance = min(
                _distance_to_prototypes(
                    recording.sequence,
                    [item.sequence for item in prototypes[other_word]],
                )
                for other_word in words
                if other_word != word
            )
            positive_distances.append(target_distance)
            positive_margins.append(competitor_distance - target_distance)

        negative_distances = [
            _distance_to_prototypes(recording.sequence, target_prototypes)
            for other_word in words
            if other_word != word
            for recording in grouped[other_word]
            if recording.dataset_split == "validation"
        ]
        positive_accept = _quantile(positive_distances, 0.90)
        negative_low = (
            _quantile(negative_distances, 0.10)
            if negative_distances
            else positive_accept
        )
        accept_threshold = min(positive_accept, negative_low)
        reject_threshold = max(
            accept_threshold + 0.02,
            _quantile(positive_distances, 0.97),
        )
        # Require an observable contrastive advantage. If the validation set
        # cannot demonstrate one, correctness remains deliberately undecidable.
        margin_threshold = max(0.01, _quantile(positive_margins, 0.10))
        statistics: dict[str, float | int | None] = {
            "validation_positive_count": len(positive_distances),
            "validation_negative_count": len(negative_distances),
            "validation_positive_p90": positive_accept,
            "validation_negative_p10": negative_low,
            "validation_margin_p10": _quantile(positive_margins, 0.10),
            "validation_margin_median": float(np.median(positive_margins)),
        }
        store_temporal_benchmark(
            word=word,
            language="en",
            locale=locale,
            task=task,
            prototype_recording_ids=[
                item.recording_id for item in prototypes[word]
            ],
            prototypes=target_prototypes,
            accept_threshold=accept_threshold,
            reject_threshold=reject_threshold,
            margin_threshold=margin_threshold,
            distance_statistics=statistics,
            sample_count=len(training),
            speaker_count=len({item.speaker_hash for item in training}),
            extractor_version=TEMPORAL_EXTRACTOR_VERSION,
            distance_version=TEMPORAL_DISTANCE_VERSION,
            decision_version=TEMPORAL_DECISION_VERSION,
            version=version,
        )
        benchmark = find_temporal_benchmark(word, locale=locale, task=task)
        if benchmark is None:
            raise RuntimeError("Stored temporal benchmark could not be loaded")
        built.append(benchmark)
    return built


def _contrastive_decision(
    target: TemporalBenchmark,
    distances: dict[str, float],
    *,
    cleaning_status: str,
) -> ContrastiveDecision:
    """Apply target thresholds only when the target beats its competitors."""
    target_distance = distances.get(target.word)
    competitors = [
        (word, distance)
        for word, distance in distances.items()
        if word != target.word
    ]
    competitor_word, competitor_distance = (
        min(competitors, key=lambda item: item[1])
        if competitors
        else (None, None)
    )
    margin = (
        competitor_distance - target_distance
        if competitor_distance is not None and target_distance is not None
        else None
    )
    if cleaning_status == "retry_recommended":
        return ContrastiveDecision(
            DecisionStatus.RETRY,
            target.word,
            target_distance,
            competitor_word,
            competitor_distance,
            margin,
            ("CLEANING_RETRY_RECOMMENDED",),
        )
    if target_distance is None or margin is None:
        return ContrastiveDecision(
            DecisionStatus.UNDECIDABLE,
            target.word,
            target_distance,
            competitor_word,
            competitor_distance,
            margin,
            ("INVALID_EVIDENCE",),
        )
    if (
        target_distance <= target.accept_threshold
        and margin >= target.margin_threshold
    ):
        status = DecisionStatus.CORRECT
        reasons = ("TARGET_WITHIN_DTW_BENCHMARK", "CONTRASTIVE_MARGIN_OK")
    elif (
        target_distance >= target.reject_threshold
        or margin <= -target.margin_threshold
    ):
        status = DecisionStatus.INCORRECT
        reasons = (
            "TARGET_OUTSIDE_DTW_BENCHMARK"
            if target_distance >= target.reject_threshold
            else "COMPETITOR_CLOSER",
        )
    else:
        status = DecisionStatus.UNDECIDABLE
        reasons = ("INSUFFICIENT_CONTRASTIVE_MARGIN",)
    return ContrastiveDecision(
        status,
        target.word,
        target_distance,
        competitor_word,
        competitor_distance,
        margin,
        reasons,
    )


def evaluate_temporal_word_pcm(
    word: str,
    pcm_s16le: bytes,
    *,
    locale: str = "en-US",
    task: str = DEFAULT_TASK,
    session_id: str | None = None,
) -> TemporalEvaluation:
    benchmarks = list_temporal_benchmarks(locale=locale, task=task)
    by_word = {benchmark.word: benchmark for benchmark in benchmarks}
    target = by_word.get(word.casefold())
    if target is None:
        raise LookupError(f"No active temporal benchmark for {word!r}")
    try:
        cleaned, report = clean_pcm_with_report(pcm_s16le)
    except ValueError:
        report_status = "retry_recommended"
        distances: dict[str, float] = {}
    else:
        report_status = report.status
        if report_status == "retry_recommended":
            distances = {}
        else:
            sequence = extract_temporal_features(cleaned).values
            distances = {
                benchmark.word: _distance_to_prototypes(
                    sequence,
                    benchmark.prototypes,
                )
                for benchmark in benchmarks
            }
    decision = _contrastive_decision(
        target,
        distances,
        cleaning_status=report_status,
    )
    score = (
        exp(-decision.target_distance)
        if decision.target_distance is not None
        else None
    )
    attempt_id = None
    if session_id is not None:
        attempt_id = save_temporal_attempt(
            session_id=session_id,
            benchmark_id=target.id,
            decision_status=decision.status.value,
            target_distance=decision.target_distance,
            competitor_word=decision.competitor_word,
            competitor_distance=decision.competitor_distance,
            margin=decision.margin,
            score=score,
            reason_codes=decision.reason_codes,
            decision_version=decision.decision_version,
        )
    return TemporalEvaluation(
        benchmark=target,
        decision=decision,
        distances=distances,
        score=score,
        attempt_id=attempt_id,
    )
