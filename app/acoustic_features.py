"""Deterministic acoustic features for the first benchmark implementation.

The public entry point deliberately accepts the same raw audio representation
used by the streaming pipeline: mono PCM signed 16-bit little-endian at 16 kHz.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.fft import dct


FEATURE_EXTRACTOR_VERSION = "mfcc-summary-v1"
SAMPLE_RATE = 16_000
MFCC_COUNT = 13
MEL_FILTER_COUNT = 26
SUMMARY_DIMENSION = MFCC_COUNT * 2

_FRAME_LENGTH = 400  # 25 ms
_HOP_LENGTH = 160  # 10 ms
_FFT_LENGTH = 512
_PREEMPHASIS = 0.97
_EPSILON = np.finfo(np.float64).eps


@dataclass(frozen=True)
class AcousticFeatureVector:
    """Versioned, fixed-size representation suitable for persistence."""

    version: str
    values: np.ndarray

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=np.float32)
        if values.shape != (SUMMARY_DIMENSION,):
            raise ValueError(
                f"expected {SUMMARY_DIMENSION} feature values, got {values.shape}"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("feature values must all be finite")
        object.__setattr__(self, "values", values)


def _hz_to_mel(frequency_hz: np.ndarray | float) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + np.asarray(frequency_hz) / 700.0)


def _mel_to_hz(frequency_mel: np.ndarray) -> np.ndarray:
    return 700.0 * (np.power(10.0, frequency_mel / 2595.0) - 1.0)


def _mel_filterbank() -> np.ndarray:
    mel_points = np.linspace(
        _hz_to_mel(0.0),
        _hz_to_mel(SAMPLE_RATE / 2),
        MEL_FILTER_COUNT + 2,
    )
    bins = np.floor(
        (_FFT_LENGTH + 1) * _mel_to_hz(mel_points) / SAMPLE_RATE
    ).astype(int)
    bins = np.clip(bins, 0, _FFT_LENGTH // 2)

    filters = np.zeros((MEL_FILTER_COUNT, _FFT_LENGTH // 2 + 1))
    for index in range(MEL_FILTER_COUNT):
        left, center, right = bins[index : index + 3]
        if center > left:
            filters[index, left:center] = (
                np.arange(left, center) - left
            ) / (center - left)
        if right > center:
            filters[index, center:right] = (
                right - np.arange(center, right)
            ) / (right - center)
    return filters


_MEL_FILTERBANK = _mel_filterbank()


def _frame(samples: np.ndarray) -> np.ndarray:
    if samples.size < _FRAME_LENGTH:
        samples = np.pad(samples, (0, _FRAME_LENGTH - samples.size))
    remainder = (samples.size - _FRAME_LENGTH) % _HOP_LENGTH
    if remainder:
        samples = np.pad(samples, (0, _HOP_LENGTH - remainder))
    frame_count = 1 + (samples.size - _FRAME_LENGTH) // _HOP_LENGTH
    shape = (frame_count, _FRAME_LENGTH)
    strides = (samples.strides[0] * _HOP_LENGTH, samples.strides[0])
    return np.lib.stride_tricks.as_strided(
        samples, shape=shape, strides=strides, writeable=False
    ).copy()


def extract_mfcc_summary(pcm_s16le: bytes) -> AcousticFeatureVector:
    """Return MFCC mean and standard deviation for mono 16 kHz PCM.

    Empty, odd-length, and all-silent inputs are rejected rather than silently
    generating a benchmark vector. Short non-silent inputs are zero padded.
    """

    if not pcm_s16le:
        raise ValueError("PCM input must not be empty")
    if len(pcm_s16le) % 2:
        raise ValueError("PCM s16le input must contain complete 16-bit samples")

    integer_samples = np.frombuffer(pcm_s16le, dtype="<i2")
    samples = integer_samples.astype(np.float64) / 32768.0
    if not np.any(samples):
        raise ValueError("silent PCM cannot produce acoustic features")

    emphasized = np.empty_like(samples)
    emphasized[0] = samples[0]
    emphasized[1:] = samples[1:] - _PREEMPHASIS * samples[:-1]

    frames = _frame(emphasized)
    frames *= np.hamming(_FRAME_LENGTH)
    spectrum = np.fft.rfft(frames, n=_FFT_LENGTH, axis=1)
    power = np.square(np.abs(spectrum)) / _FFT_LENGTH
    mel_energy = power @ _MEL_FILTERBANK.T
    log_mel_energy = np.log(np.maximum(mel_energy, _EPSILON))
    coefficients = dct(log_mel_energy, type=2, axis=1, norm="ortho")[
        :, :MFCC_COUNT
    ]

    summary = np.concatenate(
        (np.mean(coefficients, axis=0), np.std(coefficients, axis=0))
    ).astype(np.float32)
    if not np.all(np.isfinite(summary)):
        raise ValueError("MFCC extraction produced non-finite values")
    return AcousticFeatureVector(FEATURE_EXTRACTOR_VERSION, summary)


def extract_features(
    pcm_s16le: bytes,
    *,
    sample_rate: int = SAMPLE_RATE,
) -> AcousticFeatureVector:
    """Public, replaceable feature-extraction boundary."""
    if sample_rate != SAMPLE_RATE:
        raise ValueError(f"expected {SAMPLE_RATE} Hz PCM, got {sample_rate}")
    return extract_mfcc_summary(pcm_s16le)
