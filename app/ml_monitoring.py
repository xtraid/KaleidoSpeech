"""Deterministic A/B assignment and feature-drift measurement."""

from __future__ import annotations

from hashlib import sha256

import numpy as np


def assign_variant(
    attempt_id: str, *, candidate_percentage: float
) -> str:
    if not 0 <= candidate_percentage <= 100:
        raise ValueError("candidate_percentage must be between 0 and 100")
    bucket = int.from_bytes(sha256(attempt_id.encode()).digest()[:8], "big") % 10_000
    return "candidate" if bucket < candidate_percentage * 100 else "baseline"


def population_stability_index(
    reference: np.ndarray,
    observed: np.ndarray,
    *,
    bins: int = 10,
) -> float:
    """Return PSI; values above 0.2 conventionally warrant investigation."""
    reference = np.asarray(reference, dtype=np.float64)
    observed = np.asarray(observed, dtype=np.float64)
    if reference.ndim != 1 or observed.ndim != 1 or not len(reference) or not len(observed):
        raise ValueError("PSI needs non-empty one-dimensional samples")
    if not np.isfinite(reference).all() or not np.isfinite(observed).all():
        raise ValueError("PSI samples must be finite")
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return 0.0 if np.allclose(reference[0], observed) else float("inf")
    edges[0], edges[-1] = -np.inf, np.inf
    expected = np.histogram(reference, bins=edges)[0] / len(reference)
    actual = np.histogram(observed, bins=edges)[0] / len(observed)
    expected = np.maximum(expected, 1e-6)
    actual = np.maximum(actual, 1e-6)
    return float(np.sum((actual - expected) * np.log(actual / expected)))
