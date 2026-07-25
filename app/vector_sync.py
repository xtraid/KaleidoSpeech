"""Synchronize authoritative SQLite acoustic summaries into Redis Search."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json

import numpy as np

from app.acoustic_repository import feature_matrix, list_recordings
from app.vector_index import ensure_index, nearest, upsert_embedding


NORMALIZATION_KEY = "acoustic:embedding:normalization:v1"
SYNC_VERSION = "sqlite-mfcc-summary-hnsw-v1"


def normalization_key(locale: str) -> str:
    normalized = locale.strip().casefold()
    if not normalized:
        raise ValueError("locale cannot be empty")
    return f"{NORMALIZATION_KEY}:{normalized}"


@dataclass(frozen=True)
class VectorSyncReport:
    indexed: int
    locale: str
    dimension: int
    words: tuple[str, ...]
    version: str = SYNC_VERSION

    def as_dict(self) -> dict:
        payload = asdict(self)
        payload["words"] = list(self.words)
        return payload


def sync_recording_embeddings(redis_client, *, locale: str = "en-US") -> VectorSyncReport:
    recordings = list_recordings(locale=locale, accepted_only=True)
    if not recordings:
        raise ValueError(f"No accepted acoustic recordings for {locale}")
    matrix = feature_matrix(recordings).astype(np.float32)
    center = np.mean(matrix, axis=0)
    scale = np.std(matrix, axis=0)
    scale = np.where(scale < 1e-6, 1.0, scale).astype(np.float32)
    normalized = ((matrix - center) / scale).astype(np.float32)

    ensure_index(redis_client)
    pipeline = redis_client.pipeline(transaction=False)
    for recording, embedding in zip(recordings, normalized, strict=True):
        upsert_embedding(
            pipeline,
            recording.id,
            embedding,
            word=recording.word,
            locale=locale,
            split=recording.dataset_split,
        )
    pipeline.set(
        normalization_key(locale),
        json.dumps(
            {
                "center": center.tolist(),
                "scale": scale.tolist(),
                "dimension": int(matrix.shape[1]),
                "locale": locale,
                "version": SYNC_VERSION,
            },
            sort_keys=True,
        ),
    )
    pipeline.execute()
    return VectorSyncReport(
        indexed=len(recordings),
        locale=locale,
        dimension=int(matrix.shape[1]),
        words=tuple(sorted({recording.word for recording in recordings})),
    )


def query_recording_neighbors(
    redis_client, recording_id: int, *, locale: str = "en-US", k: int = 5,
) -> list[dict]:
    metadata = redis_client.get(normalization_key(locale))
    if metadata is None:
        raise LookupError("Vector normalization metadata is missing; run sync first")
    if isinstance(metadata, bytes):
        metadata = metadata.decode()
    normalization = json.loads(metadata)
    if normalization.get("locale") != locale:
        raise ValueError("Stored vector normalization uses a different locale")

    recordings = list_recordings(locale=locale, accepted_only=True)
    recording = next((item for item in recordings if item.id == recording_id), None)
    if recording is None or recording.feature is None:
        raise LookupError(f"Recording {recording_id} is unavailable")
    center = np.asarray(normalization["center"], dtype=np.float32)
    scale = np.asarray(normalization["scale"], dtype=np.float32)
    embedding = (
        np.asarray(recording.feature, dtype=np.float32) - center
    ) / scale
    return nearest(redis_client, embedding, locale=locale, k=k)
