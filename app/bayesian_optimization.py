"""Deterministic Gaussian-process search for joint decision thresholds."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from scipy.stats import norm


@dataclass(frozen=True)
class BayesianSearchResult:
    parameters: tuple[float, ...]
    objective: float
    evaluations: int


def _kernel(
    left: np.ndarray, right: np.ndarray, length_scale: np.ndarray
) -> np.ndarray:
    delta = (left[:, None, :] - right[None, :, :]) / length_scale
    return np.exp(-0.5 * np.sum(delta * delta, axis=2))


def bayesian_minimize(
    objective: Callable[[tuple[float, ...]], float],
    bounds: Sequence[tuple[float, float]],
    *,
    initial_points: int = 8,
    iterations: int = 24,
    candidates_per_iteration: int = 512,
    seed: int = 0,
) -> BayesianSearchResult:
    """Minimize an expensive bounded function using GP expected improvement."""
    if not bounds or any(not low < high for low, high in bounds):
        raise ValueError("Every parameter needs an ordered finite bound")
    if initial_points < 2 or iterations < 0 or candidates_per_iteration < 1:
        raise ValueError("Invalid Bayesian search budget")
    limits = np.asarray(bounds, dtype=np.float64)
    if not np.isfinite(limits).all():
        raise ValueError("Bounds must be finite")
    generator = np.random.default_rng(seed)
    lower, upper = limits[:, 0], limits[:, 1]
    points = generator.uniform(lower, upper, size=(initial_points, len(bounds)))
    values = np.asarray(
        [objective(tuple(point)) for point in points], dtype=np.float64
    )
    if not np.isfinite(values).all():
        raise ValueError("Objective must return finite values")
    length_scale = np.maximum((upper - lower) / 5, 1e-12)

    for _ in range(iterations):
        candidates = generator.uniform(
            lower, upper, size=(candidates_per_iteration, len(bounds))
        )
        covariance = _kernel(points, points, length_scale)
        covariance.flat[:: len(points) + 1] += 1e-6
        inverse_values = np.linalg.solve(covariance, values)
        cross = _kernel(candidates, points, length_scale)
        mean = cross @ inverse_values
        solved = np.linalg.solve(covariance, cross.T)
        variance = np.maximum(1 - np.sum(cross * solved.T, axis=1), 1e-12)
        deviation = np.sqrt(variance)
        improvement = float(np.min(values)) - mean
        z_score = improvement / deviation
        expected_improvement = (
            improvement * norm.cdf(z_score) + deviation * norm.pdf(z_score)
        )
        point = candidates[int(np.argmax(expected_improvement))]
        value = float(objective(tuple(point)))
        if not np.isfinite(value):
            raise ValueError("Objective must return finite values")
        points = np.vstack((points, point))
        values = np.append(values, value)

    best = int(np.argmin(values))
    return BayesianSearchResult(
        parameters=tuple(float(value) for value in points[best]),
        objective=float(values[best]),
        evaluations=len(values),
    )
