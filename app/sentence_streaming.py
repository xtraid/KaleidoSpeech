"""Streaming alignment of a known sentence against an annotated benchmark."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.temporal_features import FEATURE_DIMENSION, extract_temporal_features


PHRASE_STREAM_VERSION = "known-phrase-online-dtw-v1"


@dataclass(frozen=True)
class SentenceUnit:
    text: str
    start_ms: int
    end_ms: int

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("sentence unit text cannot be empty")
        if self.start_ms < 0 or self.end_ms <= self.start_ms:
            raise ValueError("sentence unit timestamps must be ordered")


@dataclass(frozen=True)
class SentenceBenchmark:
    id: int
    sentence_id: str
    text: str
    language: str
    locale: str
    reference: np.ndarray
    units: tuple[SentenceUnit, ...]
    error_threshold: float
    final_threshold: float
    version: str

    def __post_init__(self) -> None:
        reference = np.asarray(self.reference, dtype=np.float32)
        if (
            reference.ndim != 2
            or not len(reference)
            or reference.shape[1] != FEATURE_DIMENSION
            or not np.all(np.isfinite(reference))
        ):
            raise ValueError("invalid sentence benchmark feature sequence")
        if not self.units:
            raise ValueError("sentence benchmark requires annotated units")
        if self.error_threshold <= 0 or self.final_threshold <= 0:
            raise ValueError("sentence thresholds must be positive")
        object.__setattr__(self, "reference", reference)


def extract_streaming_features(
    pcm_s16le: bytes,
    *,
    chunk_ms: int = 200,
    sample_rate: int = 16_000,
) -> np.ndarray:
    """Extract deterministic chunk-local features shared by build and live paths."""
    if chunk_ms <= 0 or chunk_ms % 40:
        raise ValueError("chunk_ms must be a positive multiple of 40")
    chunk_bytes = sample_rate * chunk_ms // 1_000 * 2
    if not pcm_s16le or len(pcm_s16le) % chunk_bytes:
        raise ValueError("PCM must contain complete streaming chunks")
    sequences = [
        extract_temporal_features(
            pcm_s16le[offset : offset + chunk_bytes],
            sample_rate=sample_rate,
        ).values
        for offset in range(0, len(pcm_s16le), chunk_bytes)
    ]
    return np.concatenate(sequences, axis=0)


def _cosine_costs(frame: np.ndarray, reference: np.ndarray) -> np.ndarray:
    frame_norm = float(np.linalg.norm(frame))
    reference_norms = np.linalg.norm(reference, axis=1)
    denominator = np.maximum(frame_norm * reference_norms, 1e-8)
    similarity = reference @ frame / denominator
    return 1.0 - np.clip(similarity, -1.0, 1.0)


class OnlineSentenceAligner:
    """Incremental DTW retaining one dynamic-programming row."""

    def __init__(self, benchmark: SentenceBenchmark) -> None:
        self.benchmark = benchmark
        columns = len(benchmark.reference)
        self._costs = np.full(columns + 1, np.inf, dtype=np.float64)
        self._lengths = np.zeros(columns + 1, dtype=np.int32)
        self._costs[0] = 0.0
        self.observed_frames = 0
        self.reference_index = 0
        self.path_cost = 0.0

    def push(self, features: np.ndarray) -> list[tuple[int, float]]:
        """Consume feature rows and return `(reference_index, local_cost)` pairs."""
        values = np.asarray(features, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != FEATURE_DIMENSION:
            raise ValueError("invalid streaming feature shape")
        updates: list[tuple[int, float]] = []
        for frame in values:
            local = _cosine_costs(frame, self.benchmark.reference)
            next_costs = np.full_like(self._costs, np.inf)
            next_lengths = np.zeros_like(self._lengths)
            # A live path may stay on a reference frame or advance by one.
            for column in range(1, len(next_costs)):
                choices = (
                    (self._costs[column], self._lengths[column]),
                    (self._costs[column - 1], self._lengths[column - 1]),
                )
                previous_cost, previous_length = min(
                    choices, key=lambda item: (item[0], item[1])
                )
                if np.isfinite(previous_cost):
                    next_costs[column] = previous_cost + local[column - 1]
                    next_lengths[column] = previous_length + 1
            finite = np.flatnonzero(np.isfinite(next_costs[1:])) + 1
            if not len(finite):
                raise RuntimeError("online DTW lost every alignment path")
            normalized = next_costs[finite] / np.maximum(next_lengths[finite], 1)
            best_column = int(finite[int(np.argmin(normalized))])
            self._costs = next_costs
            self._lengths = next_lengths
            self.observed_frames += 1
            self.reference_index = best_column - 1
            self.path_cost = float(next_costs[best_column] / next_lengths[best_column])
            updates.append((self.reference_index, float(local[self.reference_index])))
        return updates

    def final_distance(self) -> float:
        """Return normalized cost at the end of the complete reference."""
        column = len(self.benchmark.reference)
        length = int(self._lengths[column])
        if not length or not np.isfinite(self._costs[column]):
            return float("inf")
        return float(self._costs[column] / length)

    def unit_at(self, reference_index: int) -> tuple[int, SentenceUnit]:
        duration = max(unit.end_ms for unit in self.benchmark.units)
        position_ms = reference_index * duration / len(self.benchmark.reference)
        for index, unit in enumerate(self.benchmark.units):
            if unit.start_ms <= position_ms < unit.end_ms:
                return index, unit
        return len(self.benchmark.units) - 1, self.benchmark.units[-1]


class SentenceStreamingSession:
    """40 ms state machine producing immediate, debounced error events."""

    def __init__(
        self,
        benchmark: SentenceBenchmark,
        *,
        frame_ms: int = 40,
        feature_chunk_ms: int = 200,
        voice_threshold: float = 0.012,
        error_confirmation_chunks: int = 2,
    ) -> None:
        if feature_chunk_ms % frame_ms:
            raise ValueError("feature chunk must be a multiple of frame duration")
        self.benchmark = benchmark
        self.frame_ms = frame_ms
        self.feature_chunk_ms = feature_chunk_ms
        self.voice_threshold = voice_threshold
        self.error_confirmation_chunks = error_confirmation_chunks
        self.aligner = OnlineSentenceAligner(benchmark)
        self._frames: list[bytes] = []
        self._chunk: list[bytes] = []
        self._bad_chunks = 0
        self._error_unit: int | None = None
        self._error_start_ms: int | None = None

    def push_frame(self, frame: bytes) -> list[dict]:
        if len(frame) != 1_280:
            raise ValueError("expected one mono 16 kHz 40 ms PCM frame")
        self._frames.append(frame)
        self._chunk.append(frame)
        elapsed_ms = len(self._frames) * self.frame_ms
        samples = np.frombuffer(frame, dtype="<i2").astype(np.float32) / 32768.0
        rms = float(np.sqrt(np.mean(np.square(samples))))
        events = [{
            "type": "sentence.listening",
            "elapsed_ms": elapsed_ms,
            "voice_activity": rms >= self.voice_threshold,
        }]
        required = self.feature_chunk_ms // self.frame_ms
        if len(self._chunk) < required:
            return events

        pcm = b"".join(self._chunk)
        self._chunk.clear()
        chunk_samples = (
            np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        )
        chunk_rms = float(np.sqrt(np.mean(np.square(chunk_samples))))
        chunk_voice = chunk_rms >= self.voice_threshold
        if not chunk_voice and self.aligner.observed_frames == 0:
            return events
        features = extract_streaming_features(
            pcm, chunk_ms=self.feature_chunk_ms
        )
        updates = self.aligner.push(features)
        reference_index, local_cost = updates[-1]
        unit_index, unit = self.aligner.unit_at(reference_index)
        events.append({
            "type": "sentence.alignment",
            "elapsed_ms": elapsed_ms,
            "unit": unit.text,
            "unit_index": unit_index,
            "reference_position": reference_index,
            "local_distance": round(local_cost, 6),
            "chunk_voice_activity": chunk_voice,
            "chunk_rms": round(chunk_rms, 6),
        })

        if local_cost > self.benchmark.error_threshold:
            if self._error_unit != unit_index:
                self._error_unit = unit_index
                self._bad_chunks = 0
                self._error_start_ms = max(0, elapsed_ms - self.feature_chunk_ms)
            self._bad_chunks += 1
            if self._bad_chunks == self.error_confirmation_chunks:
                events.append({
                    "type": "pronunciation.error",
                    "sentence_id": self.benchmark.sentence_id,
                    "unit": unit.text,
                    "unit_index": unit_index,
                    "start_ms": self._error_start_ms,
                    "detected_at_ms": elapsed_ms,
                    "status": "INCORRECT",
                    "local_distance": round(local_cost, 6),
                    "model_version": PHRASE_STREAM_VERSION,
                })
        else:
            self._bad_chunks = 0
            self._error_unit = None
            self._error_start_ms = None
        return events

    def finish(self) -> dict:
        distance = self.aligner.final_distance()
        status = (
            "RETRY"
            if not np.isfinite(distance)
            else "CORRECT"
            if distance <= self.benchmark.final_threshold
            else "INCORRECT"
        )
        return {
            "type": "sentence.evaluated",
            "sentence_id": self.benchmark.sentence_id,
            "status": status,
            "distance": None if not np.isfinite(distance) else round(distance, 6),
            "threshold": self.benchmark.final_threshold,
            "duration_ms": len(self._frames) * self.frame_ms,
            "model_version": PHRASE_STREAM_VERSION,
        }
