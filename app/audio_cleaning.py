from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, istft, resample_poly, sosfiltfilt, stft


# info output prestabilito
PIPELINE_VERSION = "1.0.0"
TARGET_SAMPLE_RATE = 16_000
FRAME_MS = 20
PADDING_MS = 120
MAX_FILE_SIZE_BYTES = 20 * 1024 * 1024


@dataclass
class CleaningReport:
    # dichiarazione valori

    pipeline_version: str
    status: str
    input_sample_rate: int
    output_sample_rate: int
    input_channels: int
    input_duration_s: float
    output_duration_s: float
    speech_duration_s: float
    speech_ratio: float
    input_peak_dbfs: float
    output_peak_dbfs: float
    rms_dbfs: float
    voice_rms_dbfs: float
    noise_floor_dbfs: float
    snr_db: float
    input_clipped_ratio: float
    dc_offset: float
    denoise_applied: bool
    warnings: list[str]


def _rms(samples: np.ndarray) -> float:
    # calcolo intensità media dell'audio input (rms)

    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))


def _to_dbfs(value: float, floor_db: float = -120.0) -> float:

    if value <= 0 or not np.isfinite(value):
        return floor_db
    return float(max(20.0 * np.log10(value), floor_db)) # converto in dbfs "leggibile"


def _choose_mono(
    samples: np.ndarray,
) -> tuple[np.ndarray, str | None]:
    """Riduce l'audio a mono evitando cancellazioni tra canali opposti."""

    if samples.shape[1] == 1:
        return samples[:, 0].astype(np.float32), None

    left, right = samples[:, 0], samples[:, 1]
    if np.std(left) == 0 or np.std(right) == 0:
        correlation = 0.0
    else:
        correlation = float(np.corrcoef(left, right)[0, 1])

    if correlation < -0.5:
        selected = left if _rms(left) >= _rms(right) else right
        return selected.astype(np.float32), "stereo_phase_cancellation_risk"

    return np.mean(samples, axis=1, dtype=np.float32), None


def _resample( # converte freq. input a 16000hz
    samples: np.ndarray,
    source_sample_rate: int,
) -> np.ndarray:

    # applico filtro anti-alias per non creare distorsioni audio dalla normalizzazione
    if source_sample_rate == TARGET_SAMPLE_RATE:
        return samples.astype(np.float32, copy=False)

    factor = gcd(source_sample_rate, TARGET_SAMPLE_RATE)
    return resample_poly(
        samples,
        TARGET_SAMPLE_RATE // factor,
        source_sample_rate // factor,
    ).astype(np.float32)


def _high_pass(samples: np.ndarray) -> np.ndarray:
    """Rimuove DC residua e rumble sotto 70 Hz senza alterare la fase."""

    sections = butter(
        4,
        70.0,
        btype="highpass",
        fs=TARGET_SAMPLE_RATE,
        output="sos",
    )
    pad_length = 3 * (2 * len(sections) + 1)
    if len(samples) <= pad_length:
        return samples
    return sosfiltfilt(sections, samples).astype(np.float32)


def _frame_signal(
    samples: np.ndarray,
    frame_length: int,
) -> np.ndarray:
    """Divide il segnale in frame, aggiungendo zeri soltanto in coda."""

    frame_count = max(1, int(np.ceil(len(samples) / frame_length)))
    padded = np.pad(samples, (0, frame_count * frame_length - len(samples)))
    return padded.reshape(frame_count, frame_length)


def _close_short_gaps(mask: np.ndarray, max_frames: int) -> np.ndarray:
    """Colma brevi buchi interni senza eliminare le pause reali."""

    result = mask.copy()
    start = 0
    while start < len(result):
        if result[start]:
            start += 1
            continue

        end = start
        while end < len(result) and not result[end]:
            end += 1
        if start > 0 and end < len(result) and end - start <= max_frames:
            result[start:end] = True
        start = end
    return result


def _remove_short_runs(mask: np.ndarray, min_frames: int) -> np.ndarray:
    """Scarta impulsi isolati che difficilmente rappresentano voce."""

    result = mask.copy()
    start = 0
    while start < len(result):
        if not result[start]:
            start += 1
            continue

        end = start
        while end < len(result) and result[end]:
            end += 1
        if end - start < min_frames:
            result[start:end] = False
        start = end
    return result


def _detect_voice(
    samples: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    """Rileva la voce con la semplice soglia energetica della guida."""

    frame_length = round(TARGET_SAMPLE_RATE * FRAME_MS / 1000)
    frames = _frame_signal(samples, frame_length)
    frame_rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
    frame_db = np.array([_to_dbfs(value) for value in frame_rms])

    noise_floor_db = float(np.percentile(frame_db, 20))
    threshold_db = max(noise_floor_db + 9.0, -42.0)
    voice_mask = frame_db >= threshold_db
    voice_mask = _close_short_gaps(voice_mask, round(80 / FRAME_MS))
    voice_mask = _remove_short_runs(voice_mask, max(1, round(60 / FRAME_MS)))

    return voice_mask, frame_rms, frame_length, noise_floor_db


def _spectral_denoise(
    samples: np.ndarray,
    voice_start: int,
    voice_end: int,
) -> np.ndarray:
    """Applica una sottrazione spettrale moderata e non generativa."""

    noise = np.concatenate((samples[:voice_start], samples[voice_end:]))
    if len(noise) < int(0.08 * TARGET_SAMPLE_RATE):
        return samples

    segment_length = 512
    overlap = 384
    _, _, noise_spectrum = stft(
        noise,
        fs=TARGET_SAMPLE_RATE,
        nperseg=segment_length,
        noverlap=overlap,
        boundary="zeros",
    )
    noise_power = np.median(
        np.abs(noise_spectrum) ** 2,
        axis=1,
        keepdims=True,
    )

    _, _, spectrum = stft(
        samples,
        fs=TARGET_SAMPLE_RATE,
        nperseg=segment_length,
        noverlap=overlap,
        boundary="zeros",
    )
    power = np.abs(spectrum) ** 2
    cleaned_power = np.maximum(power - 0.8 * noise_power, 0.12 * power)
    gain = np.sqrt(cleaned_power / np.maximum(power, 1e-12))

    _, cleaned = istft(
        spectrum * gain,
        fs=TARGET_SAMPLE_RATE,
        nperseg=segment_length,
        noverlap=overlap,
        input_onesided=True,
    )
    if len(cleaned) < len(samples):
        cleaned = np.pad(cleaned, (0, len(samples) - len(cleaned)))
    return cleaned[: len(samples)].astype(np.float32)


def clean_wav_files(
    input_path: Path,
    output_path: Path,
    report_path: Path,
) -> CleaningReport:
    """Pulisce un WAV e produce il WAV canonico e il report JSON."""

    input_path = Path(input_path)
    output_path = Path(output_path)
    report_path = Path(report_path)

    # 1. Ingestione sicura: controlliamo limiti e contenuto decodificato.
    if input_path.stat().st_size > MAX_FILE_SIZE_BYTES:
        raise ValueError("Il file audio supera il limite di 20 MB")

    info = sf.info(input_path)
    if info.frames <= 0:
        raise ValueError("Il file audio è vuoto")
    if info.channels not in (1, 2):
        raise ValueError("Sono supportati uno o due canali")
    if not 8_000 <= info.samplerate <= 96_000:
        raise ValueError("Il sample rate non è supportato")
    if not 0.2 <= info.duration <= 20.0:
        raise ValueError("La durata dell'audio non è supportata")

    raw, source_sample_rate = sf.read(
        input_path,
        dtype="float32",
        always_2d=True,
    )
    if not np.isfinite(raw).all():
        raise ValueError("L'audio contiene valori NaN o Inf")

    # 2. verifica audio troppo alto o clipping
    input_peak = float(np.max(np.abs(raw)))
    clipped_ratio = float(np.mean(np.abs(raw) >= 0.999))
    mono, mono_warning = _choose_mono(raw) # conversione a mono
    dc_offset = float(np.mean(mono, dtype=np.float64)) # normalizzo verso lo 0db per evitare sbalzi di audio
    samples = mono - dc_offset

    warnings: list[str] = [] # lista di warnings per adattare audio corretto
    if mono_warning:
        warnings.append(mono_warning)
    if clipped_ratio > 0.01:
        warnings.append("input_clipping_retry_recommended")
    elif clipped_ratio > 0.001:
        warnings.append("input_clipping_detected")

    # 3. conversione 16kHz + filtro passa-alto
    samples = _resample(samples, source_sample_rate) # converte freq di campionamento a 16000Hz
    samples = _high_pass(samples) # filtro

    # 4. divide l'audio in blocchi da 20 ms e identifica quelli vocali (VAD)
    voice_mask, frame_rms, frame_length, noise_floor_db = _detect_voice(samples)
    # Se nessun blocco contiene voce -> errore
    if not voice_mask.any():
        raise ValueError("Non è stata rilevata alcuna regione vocale")

    # trova il primo e l'ultimo blocco vocale senza rimuovere le pause interne
    voice_frames = np.flatnonzero(voice_mask)
    voice_start = int(voice_frames[0] * frame_length)
    voice_end = int(
        min(len(samples), (voice_frames[-1] + 1) * frame_length)
    )
    # Conta soltanto i blocchi vocali per ottenere la durata effettiva della voce.
    speech_duration = float(
        np.sum(voice_mask) * frame_length / TARGET_SAMPLE_RATE
    )

    # 5. Stima il rumore dai blocchi non vocali e calcola il rapporto voce/rumore.
    voice_rms = _rms(samples[voice_start:voice_end])
    non_voice_mask = ~voice_mask
    noise_rms = (
        float(np.median(frame_rms[non_voice_mask]))
        if non_voice_mask.any()
        else 1e-8
    )
    snr_db = float(
        20.0
        * np.log10(max(voice_rms, 1e-8) / max(noise_rms, 1e-8))
    )
    # Applica un denoise se inferiore a 20 dB.
    denoise_applied = snr_db < 20.0
    if denoise_applied:
        samples = _spectral_denoise(samples, voice_start, voice_end)
    # <8 dB il rumore è troppo alto - riprova
        warnings.append("low_snr_retry_recommended")

    # 7. rimozione pezzi senza voce, lasciando 120 ms di padding per testa e coda
    padding = round(PADDING_MS * TARGET_SAMPLE_RATE / 1000)
    trim_start = max(0, voice_start - padding)
    trim_end = min(len(samples), voice_end + padding)
    cleaned = samples[trim_start:trim_end]
    local_voice_start = voice_start - trim_start
    local_voice_end = min(len(cleaned), voice_end - trim_start)

    # 8. stabilizza audio level sui -23dbfs (max -1 dBFS)
    current_voice_rms = _rms(cleaned[local_voice_start:local_voice_end])
    target_rms = 10 ** (-23.0 / 20.0)
    gain = target_rms / max(current_voice_rms, 1e-8)
    gain = float(np.clip(gain, 10 ** (-12 / 20), 10 ** (18 / 20)))
    cleaned = cleaned * gain

    peak_limit = 10 ** (-1.0 / 20.0)
    output_peak = float(np.max(np.abs(cleaned)))
    if output_peak > peak_limit:
        cleaned *= peak_limit / output_peak
    cleaned = np.clip(cleaned, -1.0, 1.0).astype(np.float32)

    # 9. genera JSON finale con report audio
    output_duration = len(cleaned) / TARGET_SAMPLE_RATE
    speech_ratio = min(
        1.0,
        speech_duration / max(output_duration, 1e-8),
    )
    retry_needed = (
        snr_db < 8.0
        or speech_duration < 0.25
        or not 0.35 <= output_duration <= 15.0
        or clipped_ratio > 0.01
    )
    if retry_needed:
        status = "retry_recommended"
    elif warnings:
        status = "accept_with_warning"
    else:
        status = "accept"

    report = CleaningReport(
        pipeline_version=PIPELINE_VERSION,
        status=status,
        input_sample_rate=source_sample_rate,
        output_sample_rate=TARGET_SAMPLE_RATE,
        input_channels=info.channels,
        input_duration_s=round(info.duration, 4),
        output_duration_s=round(output_duration, 4),
        speech_duration_s=round(speech_duration, 4),
        speech_ratio=round(speech_ratio, 4),
        input_peak_dbfs=round(_to_dbfs(input_peak), 2),
        output_peak_dbfs=round(
            _to_dbfs(float(np.max(np.abs(cleaned)))),
            2,
        ),
        rms_dbfs=round(_to_dbfs(_rms(cleaned)), 2),
        voice_rms_dbfs=round(_to_dbfs(current_voice_rms), 2),
        noise_floor_dbfs=round(noise_floor_db, 2),
        snr_db=round(snr_db, 2),
        input_clipped_ratio=round(clipped_ratio, 6),
        dc_offset=round(dc_offset, 7),
        denoise_applied=denoise_applied,
        warnings=warnings,
    )

    # 10. creazione 2 file output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write( # scrittura del WAV
        output_path,
        cleaned,
        TARGET_SAMPLE_RATE,
        subtype="PCM_16",
        format="WAV",
    )
    report_path.write_text( # scrittura del JSON
        json.dumps(asdict(report), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


# Backwards-compatible name. New code should use ``clean_wav_files`` to make
# the file-based contract explicit.
clean_audio = clean_wav_files


def main() -> None:

    parser = argparse.ArgumentParser(
        description="Cleaning conservativo di una registrazione vocale.",
    )
    parser.add_argument("input", type=Path, help="WAV da pulire")
    parser.add_argument("output", type=Path, help="WAV canonico da creare")
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("cleaning_report.json"),
        help="Percorso del report JSON",
    )
    arguments = parser.parse_args()
    clean_wav_files(arguments.input, arguments.output, arguments.report)


if __name__ == "__main__":
    main()
