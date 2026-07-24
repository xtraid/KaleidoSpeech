"""Persistence and calibration tests for the acoustic benchmark baseline."""

from __future__ import annotations

import numpy as np

from app.acoustic_benchmark import build_benchmarks, import_archives
from app.acoustic_features import FEATURE_EXTRACTOR_VERSION, SUMMARY_DIMENSION
from app.acoustic_repository import (
    find_acoustic_benchmark,
    list_recordings,
    save_acoustic_attempt,
    store_recording,
)
from app.config import get_settings
from app.dataset_ingest import AudioQuality, DatasetRecording
from app.decision import DECISION_VERSION


def test_import_limit_counts_unique_recordings_after_deduplication(
    monkeypatch, tmp_path,
) -> None:
    quality = AudioQuality(
        duration_s=0.5,
        rms_dbfs=-30.0,
        peak_dbfs=-10.0,
        clipped_ratio=0.0,
        acceptable=False,
        warnings=("test_reject",),
    )

    def recording(name: str, pcm_hash: str) -> DatasetRecording:
        return DatasetRecording(
            archive_path=tmp_path / "data.zip",
            member_path=f"bird/{name}.wav",
            word="bird",
            speaker_id=name,
            take=0,
            split="train",
            wav_sha256=f"wav-{name}",
            pcm_sha256=pcm_hash,
            pcm_s16le=bytes(16_000),
            sample_rate=16_000,
            quality=quality,
        )

    candidates = [
        recording("first", "hash-a"),
        recording("duplicate", "hash-a"),
        recording("second", "hash-b"),
    ]
    stored_hashes: list[str] = []
    monkeypatch.setattr(
        "app.acoustic_benchmark.iter_zip_recordings",
        lambda *_args, **_kwargs: iter(candidates),
    )

    def store(**values):
        stored_hashes.append(values["source_hash"])
        return len(stored_hashes)

    monkeypatch.setattr("app.acoustic_benchmark.store_recording", store)

    report = import_archives(
        [tmp_path / "data.zip"],
        limit_per_word=2,
        apply_cleaning=False,
    )

    assert stored_hashes == ["hash-a", "hash-b"]
    assert report.seen == 3
    assert report.stored == 2
    assert report.duplicates == 1
    assert report.by_word == {"bird": 2}
from scripts.init_db import main as initialize_database


def _store_feature(
    word: str,
    split: str,
    index: int,
    center: float,
) -> None:
    feature = np.full(SUMMARY_DIMENSION, center, dtype=np.float64)
    feature[index % SUMMARY_DIMENSION] += index * 0.01
    store_recording(
        word=word,
        language="en",
        locale="en-US",
        speaker_hash=f"{word}-{split}-speaker-{index}",
        take_index=0,
        source_hash=f"{word}-{split}-source-{index}",
        source_member=f"fixture/{word}/{index}.wav",
        dataset_split=split,
        duration_ms=800,
        quality_status="ACCEPT",
        quality_reasons=[],
        feature=feature,
        cleaning_version="test-cleaning",
        extractor_version=FEATURE_EXTRACTOR_VERSION,
    )


def test_build_persist_reload_and_attempt_round_trip(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("SQLITE_PATH", str(tmp_path / "acoustic.sqlite3"))
    get_settings.cache_clear()
    try:
        initialize_database()
        for word, center in (("bird", 0.0), ("happy", 4.0)):
            for index in range(8):
                _store_feature(word, "train", index, center)
            for index in range(8, 12):
                _store_feature(word, "validation", index, center)
            for index in range(12, 14):
                _store_feature(word, "test", index, center)

        built = build_benchmarks(version="test-v1")

        assert [benchmark.word for benchmark in built] == ["bird", "happy"]
        bird = find_acoustic_benchmark("BIRD")
        assert bird is not None
        assert bird.sample_count == 8
        assert bird.speaker_count == 8
        assert bird.accept_threshold < bird.reject_threshold
        assert bird.extractor_version == FEATURE_EXTRACTOR_VERSION
        assert bird.decision_version == DECISION_VERSION
        assert list_recordings(
            word="bird",
            dataset_split="test",
        )

        attempt_id = save_acoustic_attempt(
            session_id="session-test",
            benchmark_id=bird.id,
            decision_status="UNDECIDABLE",
            distance=(bird.accept_threshold + bird.reject_threshold) / 2,
            score=0.5,
            confidence=0.2,
            reason_codes=["BORDERLINE_DISTANCE"],
            decision_version=bird.decision_version,
        )
        assert attempt_id > 0

        rebuilt = build_benchmarks(version="test-v1")
        assert [item.id for item in rebuilt] == [item.id for item in built]
    finally:
        get_settings.cache_clear()
