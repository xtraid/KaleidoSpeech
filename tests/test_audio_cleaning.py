import json

import numpy as np
import pytest
import soundfile as sf

from app.audio_cleaning import PIPELINE_VERSION, clean_wav_files


def _voice_like_signal(
    sample_rate: int = 48_000,
    duration_s: float = 1.2,
) -> np.ndarray:
    """Crea un segnale ripetibile con silenzio, voce sintetica e silenzio."""

    time = np.arange(round(sample_rate * duration_s)) / sample_rate
    envelope = np.zeros_like(time)
    active = (time >= 0.2) & (time <= 1.0)
    envelope[active] = np.sin(
        np.pi * (time[active] - 0.2) / 0.8
    ) ** 0.25

    voice = (
        0.12 * np.sin(2 * np.pi * 180 * time)
        + 0.06 * np.sin(2 * np.pi * 720 * time)
        + 0.03 * np.sin(2 * np.pi * 2_200 * time)
    )
    noise = np.random.default_rng(42).normal(size=len(time))
    return (envelope * voice + 0.003 * noise).astype(np.float32)


def test_clean_audio_respects_output_contract(tmp_path) -> None:
    raw_path = tmp_path / "raw.wav"
    clean_path = tmp_path / "clean.wav"
    report_path = tmp_path / "cleaning_report.json"
    sf.write(raw_path, _voice_like_signal(), 48_000)

    report = clean_wav_files(raw_path, clean_path, report_path)

    cleaned, sample_rate = sf.read(clean_path, dtype="float32")
    file_info = sf.info(clean_path)
    saved_report = json.loads(report_path.read_text(encoding="utf-8"))

    assert sample_rate == 16_000
    assert file_info.channels == 1
    assert file_info.subtype == "PCM_16"
    assert np.isfinite(cleaned).all()
    assert np.max(np.abs(cleaned)) <= 10 ** (-0.8 / 20)
    assert 0.35 <= file_info.duration <= 15.0
    assert report.pipeline_version == PIPELINE_VERSION
    assert saved_report["status"] in {
        "accept",
        "accept_with_warning",
        "retry_recommended",
    }


def test_clean_audio_rejects_audio_without_voice(tmp_path) -> None:
    raw_path = tmp_path / "silence.wav"
    sf.write(raw_path, np.zeros(16_000, dtype=np.float32), 16_000)

    with pytest.raises(ValueError, match="regione vocale"):
        clean_wav_files(
            raw_path,
            tmp_path / "clean.wav",
            tmp_path / "report.json",
        )
