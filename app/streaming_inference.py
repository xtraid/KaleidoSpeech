"""40 ms PCM streaming with lightweight feedback and rolling DTW evidence."""

from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import math
import re
from typing import Callable

import numpy as np

from app.cleaning import clean_pcm_with_report
from app.config import get_settings
from app.i18n import message, normalize_locale
from app.sentence_composer import SentencePrompt, compose_sentence
from app.temporal_benchmark import (
    _distance_to_prototypes,
    _subsequence_distance_to_prototypes,
)
from app.temporal_features import extract_temporal_features
from app.temporal_repository import (
    find_temporal_benchmark,
    list_temporal_benchmarks,
)


STREAMING_MODEL_VERSION = "rolling-dtw-observation-v1"
DistanceProvider = Callable[[bytes], dict[str, float]]
FinalEvaluator = Callable[[bytes, str], dict]
SentenceFinalEvaluator = Callable[[bytes, str, str], dict]


def database_distance_provider(
    *, locale: str = "en-US", task: str = "word_pronunciation",
    downsample_factor: int | None = None,
    sentence_mode: bool = False,
) -> DistanceProvider:
    benchmarks = list_temporal_benchmarks(locale=locale, task=task)
    if not benchmarks:
        raise LookupError("No active temporal benchmarks")
    factor = (
        get_settings().temporal_downsample_factor
        if downsample_factor is None
        else downsample_factor
    )
    if factor < 1:
        raise ValueError("downsample_factor must be positive")

    def distances(pcm: bytes) -> dict[str, float]:
        try:
            cleaned, report = clean_pcm_with_report(pcm)
        except ValueError:
            return {}
        if report.status == "retry_recommended":
            return {}
        sequence = extract_temporal_features(cleaned).values[::factor]
        distance_function = (
            _subsequence_distance_to_prototypes
            if sentence_mode
            else _distance_to_prototypes
        )
        return {
            benchmark.word: (
                distance_function(
                    sequence,
                    (benchmark.prototypes[0][::factor],),
                    length_ratios=(1.0,),
                )
                if sentence_mode
                else distance_function(
                    sequence,
                    tuple(
                        prototype[::factor]
                        for prototype in benchmark.prototypes
                    ),
                )
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
    phonetic_refresh_ms: int = 800
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
        final_evaluator: FinalEvaluator | None = None,
        sentence_final_evaluator: SentenceFinalEvaluator | None = None,
        seed: int | str | None = None,
        config: StreamingConfig | None = None,
        locale: str = "en",
        expected_text: str | None = None,
        sentence_accept_threshold: float | None = None,
    ) -> None:
        self.config = config or StreamingConfig(
            voice_rms_threshold=get_settings().voice_rms_threshold
        )
        self.prompt: SentencePrompt = compose_sentence(target_words, seed=seed)
        self.expected_text = expected_text.strip() if expected_text else None
        self._sentence_tokens = (
            re.findall(r"[A-Za-z']+", self.expected_text)
            if self.expected_text
            else []
        )
        if self.expected_text and sentence_accept_threshold is None:
            target = (
                self.prompt.target_words[0]
                if len(self.prompt.target_words) == 1
                else None
            )
            benchmark = (
                find_temporal_benchmark(target, locale=locale)
                if target is not None
                else None
            )
            sentence_accept_threshold = (
                benchmark.accept_threshold if benchmark is not None else None
            )
        self._sentence_accept_threshold = sentence_accept_threshold
        self._distance_provider = distance_provider or database_distance_provider(
            sentence_mode=self.expected_text is not None
        )
        self._final_evaluator = final_evaluator
        self._sentence_final_evaluator = sentence_final_evaluator
        self.locale = normalize_locale(locale)
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
        self._sentence_pending: Future[dict] | None = None
        self._sentence_executor = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="stream-phonetic")
            if (
                self.config.background_inference
                and self.expected_text
                and self._sentence_final_evaluator is not None
            )
            else None
        )
        self._emitted_sentence_words: set[int] = set()

    @property
    def pcm(self) -> bytes:
        return b"".join(self._frames)

    def start_event(self) -> dict:
        return {
            "type": "session.prompt",
            "prompt": self.expected_text or self.prompt.text,
            "expected_text": self.expected_text,
            "mode": "sentence_pronunciation" if self.expected_text else "word_pronunciation",
            "target_words": list(self.prompt.target_words),
            "frame_ms": self.config.frame_ms,
            "sample_rate": self.config.sample_rate,
            "encoding": "pcm_s16le",
            "model_version": STREAMING_MODEL_VERSION,
            "locale": self.locale,
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
        word_updates: list[dict] = []
        if self._sentence_pending is not None and self._sentence_pending.done():
            try:
                sentence_result = self._sentence_pending.result()
                stable_before_ms = max(0, elapsed - 240)
                for item in sentence_result.get("words", []):
                    index = int(item["index"])
                    phone_ends = [
                        int(phone["end_ms"]) for phone in item.get("phones", [])
                    ]
                    if (
                        index not in self._emitted_sentence_words
                        and phone_ends
                        and max(phone_ends) <= stable_before_ms
                    ):
                        word_updates.append(item)
                        self._emitted_sentence_words.add(index)
                        # Preserve spoken order and give the client one stable
                        # visual transition per event instead of collapsing a
                        # whole prefix into a single render tick.
                        break
            except Exception as error:
                self._model_error = f"{type(error).__name__}: {error}"
            self._sentence_pending = None
        phonetic_refresh_frames = max(
            1, self.config.phonetic_refresh_ms // self.config.frame_ms
        )
        if (
            self._sentence_final_evaluator is not None
            and self._sentence_pending is None
            and elapsed >= self.config.phonetic_refresh_ms
            and len(self._frames) % phonetic_refresh_frames == 0
        ):
            arguments = (
                self.pcm,
                self.expected_text,
                self.prompt.target_words[0],
            )
            if self._sentence_executor is None:
                self._sentence_pending = Future()
                try:
                    self._sentence_pending.set_result(
                        self._sentence_final_evaluator(*arguments)
                    )
                except Exception as error:
                    self._sentence_pending.set_exception(error)
            else:
                self._sentence_pending = self._sentence_executor.submit(
                    self._sentence_final_evaluator,
                    *arguments,
                )
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
        word_identity = None
        if self._latest_distances:
            ordered = sorted(
                self._latest_distances.items(), key=lambda item: item[1]
            )
            logits = -np.asarray([distance for _, distance in ordered])
            probabilities = np.exp(logits - np.max(logits))
            probabilities /= probabilities.sum()
            by_word = {
                word: round(float(probability), 6)
                for (word, _), probability in zip(ordered, probabilities)
            }
            word_identity = {
                "recognized_word": closest,
                "confidence": by_word[closest],
                "probabilities": by_word,
                "model_version": "closed-vocabulary-dtw-gate-v1",
            }
        sentence_words: list[dict[str, object]] = []
        current_word_index: int | None = None
        if self.expected_text:
            tokens = self._sentence_tokens
            sentence_words = [
                {
                    "text": token,
                    "index": index,
                    "status": "PENDING",
                }
                for index, token in enumerate(tokens)
            ]
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
            "word_identity": word_identity,
            "expected_text": self.expected_text,
            "current_word_index": current_word_index,
            "word_acquired_index": None,
            "words": sentence_words,
            "word_updates": word_updates,
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
        if self._sentence_executor is not None:
            # The model instance is shared with the final evaluator. Never run
            # a final pass concurrently with an unfinished rolling pass.
            self._sentence_executor.shutdown(wait=True, cancel_futures=True)
        duration = len(self._frames) * self.config.frame_ms
        voice_ratio = self._voice_frames / len(self._frames) if self._frames else 0
        if (
            duration >= 400
            and voice_ratio >= 0.1
            and self.expected_text
            and self._sentence_final_evaluator is not None
            and len(self.prompt.target_words) == 1
        ):
            try:
                result = self._sentence_final_evaluator(
                    self.pcm,
                    self.expected_text,
                    self.prompt.target_words[0],
                )
            except Exception as error:
                result = {
                    "status": "UNDECIDABLE",
                    "reason_codes": ["PHONETIC_BACKEND_ERROR"],
                    "model_error": f"{type(error).__name__}: {error}",
                }
            return {
                "type": "stream.inference.final",
                "duration_ms": duration,
                "voice_frame_ratio": round(voice_ratio, 4),
                "expected_text": self.expected_text,
                "mode": "sentence_pronunciation",
                **result,
                "provisional": False,
                "clinical_use": False,
            }
        if (
            duration >= 400
            and voice_ratio >= 0.1
            and self._final_evaluator is not None
            and len(self.prompt.target_words) == 1
        ):
            try:
                result = self._final_evaluator(
                    self.pcm, self.prompt.target_words[0]
                )
            except Exception as error:
                result = {
                    "status": "UNDECIDABLE",
                    "reason_codes": ["PHONETIC_BACKEND_ERROR"],
                    "model_error": f"{type(error).__name__}: {error}",
                }
            return {
                "type": "stream.inference.final",
                "duration_ms": duration,
                "voice_frame_ratio": round(voice_ratio, 4),
                **result,
                "provisional": False,
                "clinical_use": False,
            }
        target_word = (
            self.prompt.target_words[0]
            if len(self.prompt.target_words) == 1
            else None
        )
        recognized_word: str | None = None
        identity_confidence = 0.0
        if duration < 400 or voice_ratio < 0.1:
            status = "RETRY"
            reason_codes = ["AUDIO_INSUFFICIENT"]
        elif target_word is None:
            status = "REVIEW_REQUIRED"
            reason_codes = ["MULTI_WORD_REVIEW_REQUIRED"]
        else:
            try:
                distances = self._distance_provider(self.pcm)
            except Exception as error:
                distances = {}
                self._model_error = f"{type(error).__name__}: {error}"
            if target_word not in distances or len(distances) < 2:
                status = "UNDECIDABLE"
                reason_codes = ["WORD_IDENTITY_UNAVAILABLE"]
            else:
                ordered = sorted(distances.items(), key=lambda item: item[1])
                recognized_word = ordered[0][0]
                logits = -np.asarray([distance for _, distance in ordered])
                probabilities = np.exp(logits - np.max(logits))
                probabilities /= probabilities.sum()
                identity_confidence = float(probabilities[0])
                margin = ordered[1][1] - ordered[0][1]
                if margin < get_settings().word_gate_mismatch_margin_threshold:
                    status = "UNDECIDABLE"
                    reason_codes = ["WORD_IDENTITY_UNCERTAIN"]
                elif (
                    self.expected_text
                    and (
                        self._sentence_accept_threshold is None
                        or distances[target_word]
                        > self._sentence_accept_threshold
                    )
                ):
                    status = "INCORRECT"
                    reason_codes = ["TARGET_WORD_NOT_DETECTED"]
                elif recognized_word != target_word:
                    status = "INCORRECT"
                    reason_codes = ["WRONG_WORD"]
                elif self.expected_text:
                    status = "REVIEW_REQUIRED"
                    reason_codes = ["PHONETIC_BACKEND_REQUIRED"]
                else:
                    status = "REVIEW_REQUIRED"
                    reason_codes = ["PHONETIC_BACKEND_DISABLED"]
        return {
            "type": "stream.inference.final",
            "duration_ms": duration,
            "voice_frame_ratio": round(voice_ratio, 4),
            "status": status,
            "reason_codes": reason_codes,
            "target_word": target_word,
            "recognized_word": recognized_word,
            "word_identity_confidence": round(identity_confidence, 6),
            "expected_text": self.expected_text,
            "mode": (
                "sentence_pronunciation"
                if self.expected_text
                else "word_pronunciation"
            ),
            "observations": {
                "last_rolling_distances": self._latest_distances,
                "model_error": self._model_error,
                "alignment_status": "not_available",
            },
            "message": (
                message(self.locale, "retry")
                if status == "RETRY"
                else message(self.locale, "review")
            ),
            "clinical_use": False,
            "provisional": self.expected_text is not None,
        }
