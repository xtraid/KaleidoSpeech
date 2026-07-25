import pytest

from app.resilience import CircuitBreaker, CircuitOpenError, call_with_retry


def test_retry_recovers_and_resets_breaker() -> None:
    calls = 0

    def operation() -> str:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise OSError("temporary")
        return "ok"

    breaker = CircuitBreaker(failure_threshold=3)
    assert call_with_retry(
        operation,
        breaker=breaker,
        retry_on=(OSError,),
        attempts=3,
        initial_delay_seconds=0,
    ) == "ok"
    assert calls == 3


def test_breaker_opens_after_repeated_failures() -> None:
    breaker = CircuitBreaker(failure_threshold=1, recovery_timeout_seconds=10)
    breaker.failure(now=1)
    with pytest.raises(CircuitOpenError):
        breaker.before_call(now=2)
    breaker.before_call(now=11)
