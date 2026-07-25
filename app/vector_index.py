"""Optional Redis 8 Search index for 26-dimensional acoustic embeddings."""

from __future__ import annotations

import numpy as np


INDEX_NAME = "idx:acoustic_embeddings"
KEY_PREFIX = "acoustic:embedding:"
DIMENSION = 26


def _tag_value(value: str) -> str:
    """Escape a value embedded in a RediSearch TAG query."""
    if not value:
        raise ValueError("TAG filter value cannot be empty")
    special = set(r""",.<>{}[]"'`:;!@#$%^&*()-+=~|\/? """)
    return "".join(f"\\{character}" if character in special else character
                   for character in value)


def ensure_index(client) -> None:
    try:
        client.execute_command("FT.INFO", INDEX_NAME)
        return
    except Exception as error:
        if "unknown command" in str(error).casefold():
            raise RuntimeError(
                "Redis Search is unavailable; Redis Open Source 8 is required"
            ) from error
    try:
        client.execute_command(
            "FT.CREATE", INDEX_NAME, "ON", "HASH", "PREFIX", 1, KEY_PREFIX,
            "SCHEMA", "word", "TAG", "locale", "TAG", "split", "TAG",
            "vector", "VECTOR", "HNSW", 6, "TYPE", "FLOAT32",
            "DIM", DIMENSION, "DISTANCE_METRIC", "COSINE",
        )
    except Exception as error:
        if "index already exists" not in str(error).casefold():
            raise


def upsert_embedding(
    client, recording_id: int, embedding: list[float] | np.ndarray, *,
    word: str, locale: str, split: str,
) -> str:
    values = np.asarray(embedding, dtype="<f4")
    if values.shape != (DIMENSION,) or not np.all(np.isfinite(values)):
        raise ValueError(f"Embedding must contain {DIMENSION} finite values")
    key = f"{KEY_PREFIX}{recording_id}"
    client.hset(
        key,
        mapping={
            "word": word.casefold(),
            "locale": locale,
            "split": split,
            "vector": values.tobytes(),
        },
    )
    return key


def nearest(
    client,
    embedding: list[float] | np.ndarray,
    *,
    locale: str,
    k: int = 5,
) -> list[dict]:
    values = np.asarray(embedding, dtype="<f4")
    if values.shape != (DIMENSION,):
        raise ValueError(f"Embedding must contain {DIMENSION} values")
    if k <= 0 or k > 1_000:
        raise ValueError("k must be between 1 and 1000")
    response = client.execute_command(
        "FT.SEARCH", INDEX_NAME,
        (
            f"(@locale:{{{_tag_value(locale)}}})"
            f"=>[KNN {int(k)} @vector $query AS distance]"
        ),
        "PARAMS", 2, "query", values.tobytes(),
        "SORTBY", "distance", "RETURN", 3, "word", "locale", "distance",
        "DIALECT", 2,
    )
    results: list[dict] = []
    for index in range(1, len(response), 2):
        fields = response[index + 1]
        decoded = {
            (key.decode() if isinstance(key, bytes) else key):
            (value.decode() if isinstance(value, bytes) else value)
            for key, value in zip(fields[::2], fields[1::2], strict=True)
        }
        decoded["key"] = (
            response[index].decode()
            if isinstance(response[index], bytes) else response[index]
        )
        # FLOAT32 cosine kernels may return a tiny negative value for an
        # identical vector. Keep the public distance in its mathematical range.
        decoded["distance"] = min(2.0, max(0.0, float(decoded["distance"])))
        results.append(decoded)
    return results
