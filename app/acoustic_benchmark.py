"""Offline benchmark construction and online word evaluation."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from math import exp
from pathlib import Path
from typing import Sequence

import numpy as np

from app.acoustic_features import FEATURE_EXTRACTOR_VERSION, extract_features
from app.acoustic_repository import (
    AcousticBenchmark,
    StoredRecording,
    feature_matrix,
    find_acoustic_benchmark,
    list_recordings,
    save_acoustic_attempt,
    store_acoustic_benchmark,
    store_recording,
)
from app.audio_cleaning import PIPELINE_VERSION
from app.cleaning import clean_pcm_with_report
from app.dataset_ingest import DatasetRecording, iter_zip_recordings
from app.decision import DECISION_VERSION, DecisionResult, decide
from app.temporal_features import (
    TEMPORAL_DISTANCE_VERSION,
    TEMPORAL_EXTRACTOR_VERSION,
    extract_temporal_features,
)
from app.temporal_repository import store_temporal_feature


BENCHMARK_ALGORITHM_VERSION = "robust-prototypes-v1"
DEFAULT_TASK = "word_pronunciation"


@dataclass(frozen=True)
class ImportSummary:
    seen: int
    stored: int
    duplicates: int
    accepted: int
    rejected: int
    retry_recommended: int
    by_word: dict[str, int]


@dataclass(frozen=True)
class Evaluation:
    benchmark: AcousticBenchmark
    decision: DecisionResult
    score: float | None
    confidence: float | None
    attempt_id: int | None = None


def _recording_status(recording: DatasetRecording) -> tuple[str, list[str]]:
    if recording.quality.acceptable:
        return "ACCEPT", []
    return "REJECT", list(recording.quality.warnings)


def import_archives(
    archive_paths: Sequence[Path],
    *,
    locale: str = "en-US",
    limit_per_word: int | None = None,
    apply_cleaning: bool = True,
    speaker_salt: bytes | None = None,
) -> ImportSummary:
    """Process archive members into derived SQLite rows without extracting them."""
    if limit_per_word is not None and limit_per_word <= 0:
        raise ValueError("limit_per_word must be positive")

    iterator_options = {}
    if speaker_salt is not None:
        iterator_options["speaker_salt"] = speaker_salt
    counts: Counter[str] = Counter()
    source_hashes: set[str] = set()
    seen = stored = duplicates = accepted = rejected = retries = 0

    for recording in iter_zip_recordings(
        list(archive_paths),
        **iterator_options,
    ):
        seen += 1
        # Deduplicate before applying the per-word limit: repeated archive
        # members must not consume capacity intended for unique recordings.
        if recording.pcm_sha256 in source_hashes:
            duplicates += 1
            continue
        if (
            limit_per_word is not None
            and counts[recording.word] >= limit_per_word
        ):
            continue
        source_hashes.add(recording.pcm_sha256)
        counts[recording.word] += 1

        quality_status, reasons = _recording_status(recording)
        feature_values: list[float] | None = None
        if quality_status == "ACCEPT":
            pcm = recording.pcm_s16le
            if apply_cleaning:
                try:
                    pcm, cleaning_report = clean_pcm_with_report(pcm)
                except ValueError as error:
                    quality_status = "REJECT"
                    reasons.append(f"cleaning_error:{error}")
                else:
                    if cleaning_report.status == "retry_recommended":
                        quality_status = "RETRY"
                        reasons.extend(cleaning_report.warnings)
            if quality_status == "ACCEPT":
                try:
                    feature_values = extract_features(
                        pcm,
                        sample_rate=recording.sample_rate,
                    ).values.tolist()
                except ValueError as error:
                    quality_status = "REJECT"
                    reasons.append(f"feature_error:{error}")

        recording_id = store_recording(
            word=recording.word,
            language="en",
            locale=locale,
            speaker_hash=recording.speaker_id,
            take_index=recording.take,
            source_hash=recording.pcm_sha256,
            source_member=recording.member_path,
            dataset_split=recording.split,
            duration_ms=round(recording.quality.duration_s * 1_000),
            quality_status=quality_status,
            quality_reasons=reasons,
            feature=feature_values,
            cleaning_version=PIPELINE_VERSION if apply_cleaning else "none",
            extractor_version=FEATURE_EXTRACTOR_VERSION,
        )
        if quality_status == "ACCEPT" and feature_values is not None:
            temporal = extract_temporal_features(
                pcm,
                sample_rate=recording.sample_rate,
            )
            store_temporal_feature(
                recording_id,
                temporal.values,
                extractor_version=TEMPORAL_EXTRACTOR_VERSION,
                distance_version=TEMPORAL_DISTANCE_VERSION,
            )
        stored += 1
        if quality_status == "ACCEPT":
            accepted += 1
        elif quality_status == "RETRY":
            retries += 1
        else:
            rejected += 1

    return ImportSummary(
        seen=seen,
        stored=stored,
        duplicates=duplicates,
        accepted=accepted,
        rejected=rejected,
        retry_recommended=retries,
        by_word=dict(sorted(counts.items())),
    )


def _normalize_fit(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.median(matrix, axis=0)
    scale = np.median(np.abs(matrix - center), axis=0) * 1.4826
    standard_deviation = np.std(matrix, axis=0)
    scale = np.where(scale > 1e-6, scale, standard_deviation)
    scale = np.where(scale > 1e-6, scale, 1.0)
    return center, scale


def _normalized(
    matrix: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    result = (matrix - center) / scale
    if not np.all(np.isfinite(result)):
        raise ValueError("Feature normalization produced non-finite values")
    return result


def _distances(matrix: np.ndarray, prototypes: np.ndarray) -> np.ndarray:
    pairwise = np.linalg.norm(
        matrix[:, np.newaxis, :] - prototypes[np.newaxis, :, :],
        axis=2,
    ) / np.sqrt(matrix.shape[1])
    return np.min(pairwise, axis=1)


def _choose_prototypes(matrix: np.ndarray, count: int) -> np.ndarray:
    """Choose deterministic medoids; every prototype is an actual recording."""
    if count <= 0 or count > len(matrix):
        raise ValueError("Invalid prototype count")
    selected = [
        int(np.argmin(np.linalg.norm(matrix - np.mean(matrix, axis=0), axis=1)))
    ]
    while len(selected) < count:
        current = matrix[selected]
        nearest = _distances(matrix, current)
        nearest[selected] = -1
        selected.append(int(np.argmax(nearest)))

    assignments = np.argmin(
        np.linalg.norm(
            matrix[:, np.newaxis, :] - matrix[selected][np.newaxis, :, :],
            axis=2,
        ),
        axis=1,
    )
    medoids: list[np.ndarray] = []
    for cluster in range(count):
        members = matrix[assignments == cluster]
        if not len(members):
            medoids.append(matrix[selected[cluster]])
            continue
        centroid = np.mean(members, axis=0)
        medoids.append(
            members[int(np.argmin(np.linalg.norm(members - centroid, axis=1)))]
        )
    return np.asarray(medoids)


def _quantile(values: np.ndarray, probability: float) -> float:
    if not len(values):
        raise ValueError("Cannot calibrate from an empty distance set")
    return float(np.quantile(values, probability))


def build_benchmarks(
    *,
    locale: str = "en-US",
    task: str = DEFAULT_TASK,
    version: str = "1.0.0",
    max_prototypes: int = 3,
) -> list[AcousticBenchmark]:
    """Build robust prototypes from train speakers and thresholds from validation."""
    all_recordings = list_recordings(locale=locale)
    grouped: dict[str, list[StoredRecording]] = defaultdict(list)
    for recording in all_recordings:
        grouped[recording.word].append(recording)
    if len(grouped) < 2:
        raise ValueError("At least two words are required for provisional negatives")

    built: list[AcousticBenchmark] = []
    for word in sorted(grouped):
        training = [
            recording
            for recording in grouped[word]
            if recording.dataset_split == "train"
        ]
        validation = [
            recording
            for recording in grouped[word]
            if recording.dataset_split == "validation"
        ]
        if len(training) < 2:
            raise ValueError(f"Word {word!r} has fewer than two training samples")

        training_matrix = feature_matrix(training)
        center, scale = _normalize_fit(training_matrix)
        normalized_training = _normalized(training_matrix, center, scale)
        prototype_count = min(
            max_prototypes,
            max(1, len({item.speaker_hash for item in training}) // 20),
            len(training),
        )
        prototypes = _choose_prototypes(normalized_training, prototype_count)

        positive_source = validation or training
        positive = _distances(
            _normalized(feature_matrix(positive_source), center, scale),
            prototypes,
        )
        negative_rows = [
            recording
            for other_word, recordings in grouped.items()
            if other_word != word
            for recording in recordings
            if recording.dataset_split == "validation"
        ]
        negatives = (
            _distances(
                _normalized(feature_matrix(negative_rows), center, scale),
                prototypes,
            )
            if negative_rows
            else np.asarray([], dtype=np.float64)
        )

        positive_accept = _quantile(positive, 0.90)
        positive_high = _quantile(positive, 0.97)
        if len(negatives):
            negative_low = _quantile(negatives, 0.10)
            # Accept only where both constraints agree: most genuine
            # validation readings are close and no more than roughly 10% of
            # provisional cross-word negatives enter the accept region.
            accept_threshold = min(positive_accept, negative_low)
            reject_threshold = max(accept_threshold + 0.05, positive_high)
        else:
            negative_low = None
            accept_threshold = positive_accept
            reject_threshold = max(accept_threshold + 0.05, positive_high)

        statistics: dict[str, float | int | None] = {
            "train_distance_median": float(
                np.median(_distances(normalized_training, prototypes))
            ),
            "accept_threshold": accept_threshold,
            "validation_positive_p90": positive_accept,
            "validation_positive_p97": positive_high,
            "validation_negative_p10": negative_low,
            "validation_positive_count": len(positive),
            "validation_negative_count": len(negatives),
        }
        benchmark_id = store_acoustic_benchmark(
            word=word,
            language="en",
            locale=locale,
            task=task,
            prototypes=prototypes,
            feature_center=center,
            feature_scale=scale,
            distance_statistics=statistics,
            accept_threshold=accept_threshold,
            reject_threshold=reject_threshold,
            sample_count=len(training),
            speaker_count=len({item.speaker_hash for item in training}),
            cleaning_version=training[0].cleaning_version,
            extractor_version=training[0].extractor_version,
            decision_version=DECISION_VERSION,
            version=version,
        )
        benchmark = find_acoustic_benchmark(word, locale=locale, task=task)
        if benchmark is None or benchmark.id != benchmark_id:
            raise RuntimeError("Stored benchmark could not be reloaded")
        built.append(benchmark)
    return built


def evaluate_word_pcm(
    word: str,
    pcm_s16le: bytes,
    *,
    locale: str = "en-US",
    task: str = DEFAULT_TASK,
    session_id: str | None = None,
) -> Evaluation:
    """Clean, extract, compare, decide, and optionally persist one word."""
    benchmark = find_acoustic_benchmark(word, locale=locale, task=task)
    if benchmark is None:
        raise LookupError(f"No active acoustic benchmark for {word!r}")

    try:
        cleaned, report = clean_pcm_with_report(pcm_s16le)
    except ValueError:
        cleaning_status = "retry_recommended"
        distance = None
    else:
        cleaning_status = report.status
        if cleaning_status == "retry_recommended":
            distance = None
        else:
            feature = extract_features(cleaned).values.astype(np.float64)
            center = np.asarray(benchmark.feature_center, dtype=np.float64)
            scale = np.asarray(benchmark.feature_scale, dtype=np.float64)
            prototypes = np.asarray(benchmark.prototypes, dtype=np.float64)
            distance = float(
                _distances(
                    _normalized(feature[np.newaxis, :], center, scale),
                    prototypes,
                )[0]
            )

    decision = decide(
        distance,
        accept_threshold=benchmark.accept_threshold,
        reject_threshold=benchmark.reject_threshold,
        cleaning_status=cleaning_status,
        decision_version=benchmark.decision_version,
    )
    score = exp(-distance) if distance is not None else None
    if distance is None:
        confidence = None
    elif distance <= benchmark.accept_threshold:
        confidence = min(1.0, benchmark.accept_threshold / max(distance, 1e-9))
    elif distance >= benchmark.reject_threshold:
        confidence = min(1.0, distance / benchmark.reject_threshold - 1.0)
    else:
        width = benchmark.reject_threshold - benchmark.accept_threshold
        midpoint = (benchmark.accept_threshold + benchmark.reject_threshold) / 2
        confidence = abs(distance - midpoint) / width

    attempt_id = None
    if session_id is not None:
        attempt_id = save_acoustic_attempt(
            session_id=session_id,
            benchmark_id=benchmark.id,
            decision_status=decision.status.value,
            distance=distance,
            score=score,
            confidence=confidence,
            reason_codes=[reason.value for reason in decision.reason_codes],
            decision_version=decision.decision_version,
        )
    return Evaluation(
        benchmark=benchmark,
        decision=decision,
        score=score,
        confidence=confidence,
        attempt_id=attempt_id,
    )
