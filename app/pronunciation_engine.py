"""Pronunciation analysis domain service."""

import numpy as np


def score(audio: bytes) -> float:
    """Return a placeholder normalized energy score for an audio chunk."""
    samples = np.frombuffer(audio, dtype=np.float32)
    if samples.size == 0:
        return 0.0
    return float(np.clip(np.sqrt(np.mean(np.square(samples))), 0.0, 1.0))

