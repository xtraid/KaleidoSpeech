"""Microphone capture entry point."""

import numpy as np
import sounddevice as sd


def record(seconds: float = 1.0, sample_rate: int = 16_000) -> bytes:
    frames = int(seconds * sample_rate)
    audio = sd.rec(frames, samplerate=sample_rate, channels=1, dtype="float32")
    sd.wait()
    return np.asarray(audio, dtype=np.float32).tobytes()

