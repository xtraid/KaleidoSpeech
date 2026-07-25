"""Temporal acoustic representation and Dynamic Time Warping distance."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.fft import dct


TEMPORAL_EXTRACTOR_VERSION = "mfcc-temporal-globalnorm-delta-v2"
TEMPORAL_DISTANCE_VERSION = "dtw-cosine-band-v1"
SAMPLE_RATE = 16_000
FRAME_LENGTH = 400
HOP_LENGTH = 160
FFT_LENGTH = 512
MEL_FILTER_COUNT = 26
STATIC_COEFFICIENTS = 12
FEATURE_DIMENSION = STATIC_COEFFICIENTS * 3
DTW_BAND_RATIO = 0.30
_EPSILON = 1e-8


@dataclass(frozen=True)
class TemporalFeatureSequence:
    version: str
    values: np.ndarray

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=np.float32)
        if (
            values.ndim != 2
            or values.shape[0] < 1
            or values.shape[1] != FEATURE_DIMENSION
        ):
            raise ValueError(
                "temporal features must have shape "
                f"(frames, {FEATURE_DIMENSION})"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("temporal features must be finite")
        object.__setattr__(self, "values", values)


def _hz_to_mel(value: np.ndarray | float) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + np.asarray(value) / 700.0)


def _mel_to_hz(value: np.ndarray) -> np.ndarray:
    return 700.0 * (np.power(10.0, value / 2595.0) - 1.0)


def _filterbank() -> np.ndarray:
    points = np.linspace(
        _hz_to_mel(50.0),
        _hz_to_mel(SAMPLE_RATE / 2),
        MEL_FILTER_COUNT + 2,
    )
    bins = np.floor(
        (FFT_LENGTH + 1) * _mel_to_hz(points) / SAMPLE_RATE
    ).astype(int)
    bins = np.clip(bins, 0, FFT_LENGTH // 2)
    filters = np.zeros((MEL_FILTER_COUNT, FFT_LENGTH // 2 + 1))
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


_FILTERBANK = _filterbank()


def _frames(samples: np.ndarray) -> np.ndarray:
    if samples.size < FRAME_LENGTH:
        samples = np.pad(samples, (0, FRAME_LENGTH - samples.size))
    remainder = (samples.size - FRAME_LENGTH) % HOP_LENGTH
    if remainder:
        samples = np.pad(samples, (0, HOP_LENGTH - remainder))
    count = 1 + (samples.size - FRAME_LENGTH) // HOP_LENGTH
    indexes = (
        np.arange(FRAME_LENGTH)[np.newaxis, :]
        + np.arange(count)[:, np.newaxis] * HOP_LENGTH
    )
    return samples[indexes]


def _delta(values: np.ndarray, width: int = 2) -> np.ndarray:
    padded = np.pad(values, ((width, width), (0, 0)), mode="edge")
    denominator = 2 * sum(index * index for index in range(1, width + 1))
    result = np.zeros_like(values)
    for index in range(1, width + 1):
        result += index * (
            padded[width + index : width + index + len(values)]
            - padded[width - index : width - index + len(values)]
        )
    return result / denominator


def extract_temporal_features(
    pcm_s16le: bytes,
    *,
    sample_rate: int = SAMPLE_RATE,
) -> TemporalFeatureSequence:
    """Extract CMVN-normalized MFCC trajectories with velocity and acceleration."""
    if sample_rate != SAMPLE_RATE:
        raise ValueError(f"expected {SAMPLE_RATE} Hz PCM, got {sample_rate}")
    if not pcm_s16le or len(pcm_s16le) % 2:
        raise ValueError("PCM must contain complete non-empty int16 samples")
    integer = np.frombuffer(pcm_s16le, dtype="<i2")
    samples = integer.astype(np.float64) / 32768.0
    if np.sqrt(np.mean(np.square(samples))) < 1e-5:
        raise ValueError("silent PCM cannot produce temporal features")

    emphasized = np.empty_like(samples)
    emphasized[0] = samples[0]
    emphasized[1:] = samples[1:] - 0.97 * samples[:-1]
    framed = _frames(emphasized) * np.hamming(FRAME_LENGTH)
    spectrum = np.fft.rfft(framed, n=FFT_LENGTH, axis=1)
    power = np.square(np.abs(spectrum)) / FFT_LENGTH
    mel = power @ _FILTERBANK.T
    cepstra = dct(
        np.log(np.maximum(mel, np.finfo(np.float64).eps)),
        type=2,
        axis=1,
        norm="ortho",
    )
    # Drop c0 (energy-dominated) and keep the spectral-envelope coefficients.
    static = cepstra[:, 1 : STATIC_COEFFICIENTS + 1]
    # A single utterance-level normalization removes global gain/channel
    # offsets while preserving the relative cepstral means that encode vowel
    # identity. Per-coefficient CMVN is too destructive for sub-second words.
    static = static - float(np.mean(static))
    static = static / max(float(np.std(static)), 1.0)
    velocity = _delta(static)
    acceleration = _delta(velocity)
    values = np.concatenate((static, velocity, acceleration), axis=1)
    return TemporalFeatureSequence(
        TEMPORAL_EXTRACTOR_VERSION,
        values.astype(np.float32),
    )


def _cosine_cost(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= _EPSILON:
        return float(np.linalg.norm(left - right) / np.sqrt(len(left)))
    cosine = float(np.dot(left, right) / denominator)
    return 1.0 - float(np.clip(cosine, -1.0, 1.0))


def dtw_distance(
    left: np.ndarray,
    right: np.ndarray,
    *,
    band_ratio: float = DTW_BAND_RATIO,
) -> float:
    """Return path-length-normalized DTW distance with a Sakoe-Chiba band."""
    first = np.asarray(left, dtype=np.float64)
    second = np.asarray(right, dtype=np.float64)
    if (
        first.ndim != 2
        or second.ndim != 2
        or first.shape[1] != FEATURE_DIMENSION
        or second.shape[1] != FEATURE_DIMENSION
        or not len(first)
        or not len(second)
        or not np.all(np.isfinite(first))
        or not np.all(np.isfinite(second))
    ):
        raise ValueError("DTW inputs must be finite temporal feature sequences")
    if not 0 < band_ratio <= 1:
        raise ValueError("band_ratio must be in (0, 1]")

    rows, columns = len(first), len(second)
    band = max(abs(rows - columns), int(max(rows, columns) * band_ratio), 2)
    first_norms = np.linalg.norm(first, axis=1)
    second_norms = np.linalg.norm(second, axis=1)
    denominators = first_norms[:, None] * second_norms[None, :]
    local_costs = np.empty((rows, columns), dtype=np.float64)
    valid = denominators > _EPSILON
    local_costs[valid] = 1.0 - np.clip(
        (first @ second.T)[valid] / denominators[valid],
        -1.0,
        1.0,
    )
    if not np.all(valid):
        zero_rows, zero_columns = np.nonzero(~valid)
        differences = first[zero_rows] - second[zero_columns]
        local_costs[zero_rows, zero_columns] = (
            np.linalg.norm(differences, axis=1) / np.sqrt(FEATURE_DIMENSION)
        )

    previous_costs = np.full(columns + 1, np.inf)
    previous_lengths = np.zeros(columns + 1, dtype=np.int32)
    previous_costs[0] = 0.0
    for row in range(1, rows + 1):
        current_costs = np.full(columns + 1, np.inf)
        current_lengths = np.zeros(columns + 1, dtype=np.int32)
        center = round(row * columns / rows)
        start = max(1, center - band)
        stop = min(columns, center + band)
        for column in range(start, stop + 1):
            predecessors = (
                (previous_costs[column], previous_lengths[column]),
                (current_costs[column - 1], current_lengths[column - 1]),
                (previous_costs[column - 1], previous_lengths[column - 1]),
            )
            previous_cost, previous_length = min(
                predecessors,
                key=lambda value: (value[0], value[1]),
            )
            if not np.isfinite(previous_cost):
                continue
            current_costs[column] = (
                previous_cost + local_costs[row - 1, column - 1]
            )
            current_lengths[column] = previous_length + 1
        previous_costs, current_costs = current_costs, previous_costs
        previous_lengths, current_lengths = current_lengths, previous_lengths
    path_length = int(previous_lengths[columns])
    if path_length == 0 or not np.isfinite(previous_costs[columns]):
        raise ValueError("No DTW path found inside the configured band")
    return float(previous_costs[columns] / path_length)
