"""Contracts and concrete adapters for utterance-level phoneme inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


DEFAULT_PHONEME_MODEL = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
DEFAULT_PHONEME_REVISION = "2c733782da5604684829819a5eb744c193fe9398"
PHONEME_MODEL_VERSION = f"{DEFAULT_PHONEME_MODEL}@{DEFAULT_PHONEME_REVISION}"


@dataclass(frozen=True)
class PhonemeEvidence:
    """Frame-level CTC log probabilities and their token vocabulary."""

    log_probabilities: np.ndarray
    labels: tuple[str, ...]
    blank_id: int
    frame_stride_ms: float
    model_version: str

    def __post_init__(self) -> None:
        probabilities = np.asarray(self.log_probabilities)
        if probabilities.ndim != 2 or probabilities.shape[1] != len(self.labels):
            raise ValueError("CTC evidence must have shape [frames, labels]")
        if not len(probabilities) or not 0 <= self.blank_id < len(self.labels):
            raise ValueError("CTC evidence has invalid dimensions or blank id")
        if not np.isfinite(probabilities).all() or self.frame_stride_ms <= 0:
            raise ValueError("CTC evidence contains invalid numeric values")


class PhonemeModel(Protocol):
    version: str

    def infer(self, pcm_s16le: bytes, sample_rate: int) -> PhonemeEvidence:
        """Run inference on one complete utterance, never on a 40 ms frame."""
        ...


class TransformersWav2Vec2PhonemeModel:
    """Lazy server-side adapter for the selected Hugging Face CTC checkpoint.

    Heavy dependencies and model weights are loaded only when this backend is
    explicitly constructed. This keeps the core service and tests lightweight.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_PHONEME_MODEL,
        *,
        revision: str = DEFAULT_PHONEME_REVISION,
        device: str = "cpu",
        local_files_only: bool = True,
        cache_dir: Path | str = Path("data/model-cache"),
        quantize_cpu: bool = False,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCTC, AutoProcessor
        except ImportError as error:
            raise RuntimeError(
                "Install the phonetic extra: uv sync --extra phonetic"
            ) from error

        self._torch = torch
        self._processor = AutoProcessor.from_pretrained(
            model_id, revision=revision, local_files_only=local_files_only,
            cache_dir=str(cache_dir),
        )
        self._model = AutoModelForCTC.from_pretrained(
            model_id, revision=revision, local_files_only=local_files_only,
            cache_dir=str(cache_dir),
        ).to(device)
        if quantize_cpu:
            if device != "cpu":
                raise ValueError("Dynamic quantization is supported only on CPU")
            # Linear layers dominate wav2vec2's transformer encoder. Dynamic
            # INT8 uses PyTorch's AVX2/FBGEMM kernels while convolution layers
            # remain in FP32, preserving compatibility with this checkpoint.
            self._model = torch.ao.quantization.quantize_dynamic(
                self._model,
                {torch.nn.Linear},
                dtype=torch.qint8,
                inplace=True,
            )
        self._model.eval()
        self._device = device
        self.version = f"{model_id}@{revision}"

        tokenizer = self._processor.tokenizer
        vocabulary = tokenizer.get_vocab()
        self._labels = tuple(
            token for token, _ in sorted(vocabulary.items(), key=lambda item: item[1])
        )
        blank_id = tokenizer.pad_token_id
        if blank_id is None:
            raise RuntimeError("The selected CTC tokenizer has no blank/pad token")
        self._blank_id = int(blank_id)

    def infer(self, pcm_s16le: bytes, sample_rate: int) -> PhonemeEvidence:
        if sample_rate != 16_000:
            raise ValueError("The selected phoneme model requires 16 kHz audio")
        samples = np.frombuffer(pcm_s16le, dtype="<i2").astype(np.float32) / 32768.0
        if not len(samples):
            raise ValueError("Cannot infer phonemes from empty audio")

        inputs = self._processor(
            samples, sampling_rate=sample_rate, return_tensors="pt"
        )
        input_values = inputs.input_values.to(self._device)
        with self._torch.inference_mode():
            logits = self._model(input_values).logits[0]
            log_probabilities = self._torch.log_softmax(logits, dim=-1)
        values = log_probabilities.detach().cpu().numpy().astype(np.float64)
        duration_ms = len(samples) * 1_000 / sample_rate
        return PhonemeEvidence(
            log_probabilities=values,
            labels=self._labels,
            blank_id=self._blank_id,
            frame_stride_ms=duration_ms / len(values),
            model_version=self.version,
        )


# Compatibility with the original persistence contract.
@dataclass(frozen=True)
class PronunciationResult:
    detected_phonemes: list[str]
    phoneme_confidences: list[float]
    overall_confidence: float
    score: float | None
    engine_version: str


