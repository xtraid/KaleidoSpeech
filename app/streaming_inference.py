"""40 ms PCM streaming with lightweight feedback and rolling DTW evidence."""

from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import math
from typing import Callable

import numpy as np

from app.cleaning import clean_pcm_with_report
from app.sentence_composer import SentencePrompt, compose_sentence
from app.temporal_benchmark import _distance_to_prototypes
from app.temporal_features import extract_temporal_features
from app.temporal_repository import list_temporal_benchmarks


STREAMING_MODEL_VERSION = "rolling-dtw-observation-v1"
DistanceProvider = Callable[[bytes], dict[str, float]]


def database_distance_provider(
    *, locale: str = "en-US", task: str = "word_pronunciation",
) -> DistanceProvider:
    benchmarks = list_temporal_benchmarks(locale=locale, task=task)
    if not benchmarks:
        raise LookupError("No active temporal benchmarks")

    def distances(pcm: bytes) -> dict[str, float]:
        try:
            cleaned, report = clean_pcm_with_report(pcm)
        except ValueError:
            return {}
        if report.status == "retry_recommended":
            return {}
        sequence = extract_temporal_features(cleaned).values
        return {
            benchmark.word: _distance_to_prototypes(
                sequence, benchmark.prototypes
            )
            for benchmark in benchmarks
        }

    return distances


@dataclass(frozen=True)
class StreamingConfig:
    sample_rate: int = 16_000
    frame_ms: int = 40
    minimum_window_ms: int = 400
    rolling_window_ms: int = 1_600
    acoustic_refresh_ms: int = 200
    maximum_duration_ms: int = 15_000
    voice_rms_threshold: float = 0.008
    background_inference: bool = True

    @property
    def bytes_per_frame(self) -> int:
        return self.sample_rate * self.frame_ms // 1_000 * 2


class StreamingInferenceSession:
    """Stateful stream. Outputs observations, never a clinical diagnosis."""

    def __init__(
        self, target_words: list[str] | tuple[str, ...], *,
        distance_provider: DistanceProvider | None = None,
        seed: int | str | None = None,
        config: StreamingConfig | None = None,
    ) -> None:
        self.config = config or StreamingConfig()
        self.prompt: SentencePrompt = compose_sentence(target_words, seed=seed)
        self._distance_provider = distance_provider or database_distance_provider()
        self._frames: list[bytes] = []
        max_frames = math.ceil(
            self.config.rolling_window_ms / self.config.frame_ms
        )
        self._rolling: deque[bytes] = deque(maxlen=max_frames)
        self._voice_frames = 0
        self._closed = False
        self._latest_distances: dict[str, float] = {}
        self._model_error: str | None = None
        self._executor = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="stream-dtw")
            if self.config.background_inference else None
        )
        self._pending: Future[dict[str, float]] | None = None

    @property
    def pcm(self) -> bytes:
        return b"".join(self._frames)

    def start_event(self) -> dict:
        return {
            "type": "session.prompt",
            "prompt": self.prompt.text,
            "target_words": list(self.prompt.target_words),
            "frame_ms": self.config.frame_ms,
            "sample_rate": self.config.sample_rate,
            "encoding": "pcm_s16le",
            "model_version": STREAMING_MODEL_VERSION,
        }

    def push_frame(self, frame: bytes) -> dict:
        if self._closed:
            raise RuntimeError("Streaming session is closed")
        if len(frame) != self.config.bytes_per_frame:
            raise ValueError(
                f"Expected {self.config.bytes_per_frame} bytes per 40 ms frame"
            )
        if len(self._frames) * self.config.frame_ms >= (
            self.config.maximum_duration_ms
        ):
            raise ValueError("Maximum stream duration exceeded")
        samples = np.frombuffer(frame, dtype="<i2").astype(np.float32) / 32768.0
        rms = float(np.sqrt(np.mean(np.square(samples))))
        peak = float(np.max(np.abs(samples)))
        voice = rms >= self.config.voice_rms_threshold
        self._voice_frames += int(voice)
        self._frames.append(frame)
        self._rolling.append(frame)

        elapsed = len(self._frames) * self.config.frame_ms
        rolling_duration = len(self._rolling) * self.config.frame_ms
        refresh_frames = max(
            1, self.config.acoustic_refresh_ms // self.config.frame_ms
        )
        refreshed = False
        model_scheduled = False
        if self._pending is not None and self._pending.done():
            try:
                self._latest_distances = self._pending.result()
                refreshed = True
                self._model_error = None
            except Exception as error:
                self._model_error = f"{type(error).__name__}: {error}"
            finally:
                self._pending = None
        if (
            voice
            and rolling_duration >= self.config.minimum_window_ms
            and len(self._frames) % refresh_frames == 0
            and self._pending is None
        ):
            window = b"".join(self._rolling)
            if self._executor is None:
                self._latest_distances = self._distance_provider(window)
                refreshed = True
            else:
                self._pending = self._executor.submit(
                    self._distance_provider, window
                )
                model_scheduled = True
        closest = (
            min(self._latest_distances, key=self._latest_distances.get)
            if self._latest_distances else None
        )
        return {
            "type": "stream.inference.partial",
            "sequence": len(self._frames),
            "elapsed_ms": elapsed,
            "voice_activity": voice,
            "rms": round(rms, 6),
            "peak": round(peak, 6),
            "clipping": peak >= 0.99,
            "model_refreshed": refreshed,
            "model_scheduled": model_scheduled,
            "closest_benchmark_word": closest,
            "distances": self._latest_distances,
            "model_error": self._model_error,
            "provisional": True,
            "clinical_use": False,
        }

    def finish(self) -> dict:
        if self._closed:
            raise RuntimeError("Streaming session is closed")
        self._closed = True
        if self._pending is not None and self._pending.done():
            try:
                self._latest_distances = self._pending.result()
            except Exception as error:
                self._model_error = f"{type(error).__name__}: {error}"
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
        duration = len(self._frames) * self.config.frame_ms
        voice_ratio = self._voice_frames / len(self._frames) if self._frames else 0
        status = "RETRY" if duration < 400 or voice_ratio < 0.1 else "REVIEW_REQUIRED"
        return {
            "type": "stream.inference.final",
            "duration_ms": duration,
            "voice_frame_ratio": round(voice_ratio, 4),
            "status": status,
            "observations": {
                "last_rolling_distances": self._latest_distances,
                "model_error": self._model_error,
                "alignment_status": "not_available",
            },
            "message": (
                "Audio insufficiente: ripetere."
                if status == "RETRY"
                else "Registrazione acquisita; revisione del logopedista richiesta."
            ),
            "clinical_use": False,
        }
