from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import soundfile as sf

from app.audio_cleaning import TARGET_SAMPLE_RATE, clean_audio as clean_audio_file


def clean_audio(audio: bytes) -> bytes:

    if len(audio) % np.dtype(np.float32).itemsize != 0:
        raise ValueError("I bytes audio non contengono campioni float32 completi")

    samples = np.frombuffer(audio, dtype="<f4")
    if samples.size == 0:
        raise ValueError("L'audio è vuoto")
    if not np.isfinite(samples).all():
        raise ValueError("L'audio contiene valori NaN o Inf")

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
        cleaned, sample_rate = sf.read(output_path, dtype="float32")

    if sample_rate != TARGET_SAMPLE_RATE:
        raise RuntimeError("Il cleaning ha prodotto un sample rate non valido")

    # il solver legge esplicitamente float32 little-endian
    return np.asarray(cleaned, dtype="<f4").tobytes()
