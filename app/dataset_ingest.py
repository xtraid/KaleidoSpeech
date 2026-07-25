"""Safe, deterministic ingestion of word recordings stored in ZIP archives."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import hmac
from io import BytesIO
from pathlib import Path, PurePosixPath
import re
from typing import Iterator
import wave
from zipfile import BadZipFile, ZipFile, ZipInfo

import numpy as np


INGEST_VERSION = "zip-word-recordings-v1"
EXPECTED_SAMPLE_RATE = 16_000
DEFAULT_SPEAKER_SALT = b"advx-prototype-speaker-salt-v1"
MAX_WAV_BYTES = 20 * 1024 * 1024
MAX_COMPRESSION_RATIO = 200.0
_NAME_PATTERN = re.compile(
    r"^(?P<speaker>[0-9a-fA-F]{8})_nohash_(?P<take>[0-9]+)\.wav$"
)
_WORD_PATTERN = re.compile(r"^[A-Za-z][A-Za-z'-]*$")


@dataclass(frozen=True)
class AudioQuality:
    duration_s: float
    rms_dbfs: float
    peak_dbfs: float
    clipped_ratio: float
    acceptable: bool
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class DatasetRecording:
    archive_path: Path
    member_path: str
    word: str
    speaker_id: str
    take: int
    split: str
    wav_sha256: str
    pcm_sha256: str
    pcm_s16le: bytes
    sample_rate: int
    quality: AudioQuality
    ingest_version: str = INGEST_VERSION


def pseudonymize_speaker(speaker: str, salt: bytes) -> str:
    """Create a stable project-local speaker ID without retaining the source ID."""

    if len(salt) < 16:
        raise ValueError("speaker salt must contain at least 16 bytes")
    return hmac.new(salt, speaker.lower().encode("ascii"), sha256).hexdigest()


def assign_speaker_split(
    speaker_id: str,
    *,
    seed: str = "advx-speaker-split-v1",
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
) -> str:
    """Assign a globally stable split using only the pseudonymous speaker ID."""

    if train_ratio < 0 or validation_ratio < 0:
        raise ValueError("split ratios cannot be negative")
    if train_ratio + validation_ratio > 1:
        raise ValueError("train and validation ratios cannot exceed one")
    digest = sha256(f"{seed}:{speaker_id}".encode("utf-8")).digest()
    position = int.from_bytes(digest[:8], "big") / 2**64
    if position < train_ratio:
        return "train"
    if position < train_ratio + validation_ratio:
        return "validation"
    return "test"


def deterministic_split(speaker_id: str) -> str:
    """Stable public split helper used by tests and offline tooling."""
    return assign_speaker_split(speaker_id)


def _is_ignored_member(name: str) -> bool:
    parts = PurePosixPath(name).parts
    return (
        not parts
        or "__MACOSX" in parts
        or any(part == ".DS_Store" or part.startswith("._") for part in parts)
    )


def _safe_member_path(info: ZipInfo) -> PurePosixPath:
    if "\\" in info.filename:
        raise ValueError(f"ZIP member uses unsafe separators: {info.filename!r}")
    path = PurePosixPath(info.filename)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe ZIP member path: {info.filename!r}")
    return path


def _read_bounded_member(archive: ZipFile, info: ZipInfo) -> bytes:
    if info.file_size > MAX_WAV_BYTES:
        raise ValueError(f"WAV member exceeds {MAX_WAV_BYTES} bytes")
    if info.file_size and info.compress_size == 0:
        raise ValueError("invalid compressed size")
    if info.compress_size and info.file_size / info.compress_size > MAX_COMPRESSION_RATIO:
        raise ValueError("suspicious ZIP compression ratio")

    chunks: list[bytes] = []
    total = 0
    with archive.open(info, "r") as source:
        while chunk := source.read(64 * 1024):
            total += len(chunk)
            if total > MAX_WAV_BYTES:
                raise ValueError(f"WAV member exceeds {MAX_WAV_BYTES} bytes")
            chunks.append(chunk)
    if total != info.file_size:
        raise ValueError("ZIP member size does not match its metadata")
    return b"".join(chunks)


def _decode_wav(data: bytes) -> tuple[bytes, int]:
    try:
        with wave.open(BytesIO(data), "rb") as source:
            channels = source.getnchannels()
            sample_width = source.getsampwidth()
            sample_rate = source.getframerate()
            compression = source.getcomptype()
            frame_count = source.getnframes()
            pcm = source.readframes(frame_count)
    except (EOFError, wave.Error) as error:
        raise ValueError(f"invalid WAV: {error}") from error

    if channels != 1 or sample_width != 2 or compression != "NONE":
        raise ValueError("WAV must be mono, signed 16-bit, uncompressed PCM")
    if sample_rate != EXPECTED_SAMPLE_RATE:
        raise ValueError(f"WAV sample rate must be {EXPECTED_SAMPLE_RATE} Hz")
    if len(pcm) != frame_count * sample_width:
        raise ValueError("truncated WAV sample data")
    if not pcm:
        raise ValueError("WAV contains no samples")
    return pcm, sample_rate


def assess_quality(pcm_s16le: bytes, sample_rate: int) -> AudioQuality:
    samples = np.frombuffer(pcm_s16le, dtype="<i2").astype(np.float64)
    normalized = samples / 32768.0
    duration_s = samples.size / sample_rate
    rms = float(np.sqrt(np.mean(np.square(normalized))))
    peak = float(np.max(np.abs(normalized)))
    clipped_ratio = float(np.mean(np.abs(samples) >= 32767))
    floor = np.finfo(np.float64).tiny
    rms_dbfs = float(20 * np.log10(max(rms, floor)))
    peak_dbfs = float(20 * np.log10(max(peak, floor)))

    warnings: list[str] = []
    if duration_s < 0.3:
        warnings.append("too_short")
    if duration_s > 2.0:
        warnings.append("too_long")
    if rms_dbfs < -45.0:
        warnings.append("very_low_level")
    if clipped_ratio > 0.001:
        warnings.append("clipping")
    return AudioQuality(
        duration_s=duration_s,
        rms_dbfs=rms_dbfs,
        peak_dbfs=peak_dbfs,
        clipped_ratio=clipped_ratio,
        acceptable=not warnings,
        warnings=tuple(warnings),
    )


def iter_zip_recordings(
    archive_paths: list[Path] | tuple[Path, ...],
    *,
    speaker_salt: bytes = DEFAULT_SPEAKER_SALT,
    split_seed: str = "advx-speaker-split-v1",
) -> Iterator[DatasetRecording]:
    """Yield validated recordings without extracting archive members to disk.

    Invalid WAV candidates fail closed with ``ValueError``. macOS metadata and
    non-WAV files are ignored. Callers can catch errors per archive and decide
    whether an import should be atomic.
    """

    for archive_path in archive_paths:
        try:
            archive = ZipFile(archive_path)
        except BadZipFile as error:
            raise ValueError(f"invalid ZIP archive: {archive_path}") from error
        with archive:
            for info in archive.infolist():
                path = _safe_member_path(info)
                if info.is_dir() or _is_ignored_member(info.filename):
                    continue
                if path.suffix.lower() != ".wav":
                    continue
                if len(path.parts) < 2:
                    raise ValueError(f"WAV has no word directory: {info.filename!r}")

                word = path.parent.name.lower()
                match = _NAME_PATTERN.fullmatch(path.name)
                if not _WORD_PATTERN.fullmatch(word) or match is None:
                    raise ValueError(f"unrecognized dataset path: {info.filename!r}")

                wav_data = _read_bounded_member(archive, info)
                pcm, sample_rate = _decode_wav(wav_data)
                speaker_id = pseudonymize_speaker(
                    match.group("speaker"), speaker_salt
                )
                yield DatasetRecording(
                    archive_path=Path(archive_path),
                    member_path=info.filename,
                    word=word,
                    speaker_id=speaker_id,
                    take=int(match.group("take")),
                    split=assign_speaker_split(
                        speaker_id, seed=split_seed
                    ),
                    wav_sha256=sha256(wav_data).hexdigest(),
                    pcm_sha256=sha256(pcm).hexdigest(),
                    pcm_s16le=pcm,
                    sample_rate=sample_rate,
                    quality=assess_quality(pcm, sample_rate),
                )
