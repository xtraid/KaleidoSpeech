from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import soundfile as sf

from app.audio_cleaning import TARGET_SAMPLE_RATE, clean_audio as clean_audio_file


def clean_audio(pcm_s16le: bytes) -> bytes:
    """Clean one complete mono 16 kHz PCM signed-16 little-endian window."""

    if len(pcm_s16le) % np.dtype("<i2").itemsize != 0:
        raise ValueError("I byte audio non contengono campioni int16 completi")

    integer_samples = np.frombuffer(pcm_s16le, dtype="<i2")
    samples = integer_samples.astype(np.float32) / 32768.0
    if samples.size == 0:
        raise ValueError("L'audio è vuoto")

    with TemporaryDirectory() as directory:
        temporary_directory = Path(directory)
        input_path = temporary_directory / "input.wav"
        output_path = temporary_directory / "clean.wav"
        report_path = temporary_directory / "report.json"

        sf.write(
            input_path,
            samples,
            TARGET_SAMPLE_RATE,
            subtype="FLOAT",
            format="WAV",
        )
        clean_audio_file(input_path, output_path, report_path)
        cleaned, sample_rate = sf.read(output_path, dtype="int16")

    if sample_rate != TARGET_SAMPLE_RATE:
        raise RuntimeError("Il cleaning ha prodotto un sample rate non valido")

    return np.asarray(cleaned, dtype="<i2").tobytes()
