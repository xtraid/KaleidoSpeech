"""Circuit breaker and bounded exponential retry for transient dependencies."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
import time
from typing import Callable, ParamSpec, TypeVar


P = ParamSpec("P")
T = TypeVar("T")


class CircuitOpenError(RuntimeError):
    pass


@dataclass
class CircuitBreaker:
    failure_threshold: int = 3
    recovery_timeout_seconds: float = 10.0

    def __post_init__(self) -> None:
        self._failures = 0
        self._opened_at: float | None = None
        self._lock = Lock()

    def before_call(self, *, now: float | None = None) -> None:
        current = time.monotonic() if now is None else now
        with self._lock:
            if self._opened_at is None:
                return
            if current - self._opened_at >= self.recovery_timeout_seconds:
                self._opened_at = None
                self._failures = 0
                return
            raise CircuitOpenError("Dependency circuit is open")

    def success(self) -> None:
        with self._lock:
            self._failures = 0
            self._opened_at = None

    def failure(self, *, now: float | None = None) -> None:
        current = time.monotonic() if now is None else now
        with self._lock:
            self._failures += 1
            if self._failures >= self.failure_threshold:
                self._opened_at = current


def call_with_retry(
    operation: Callable[P, T],
    *args: P.args,
    breaker: CircuitBreaker,
    retry_on: tuple[type[BaseException], ...],
    attempts: int = 3,
    initial_delay_seconds: float = 0.05,
    **kwargs: P.kwargs,
) -> T:
    if attempts < 1:
        raise ValueError("attempts must be positive")
    delay = initial_delay_seconds
    for attempt in range(attempts):
        breaker.before_call()
        try:
            result = operation(*args, **kwargs)
        except retry_on:
            breaker.failure()
            if attempt + 1 == attempts:
                raise
            time.sleep(delay)
            delay *= 2
        else:
            breaker.success()
            return result
    raise AssertionError("unreachable")
