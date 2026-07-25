from app.security import SlidingWindowRateLimiter


def test_sliding_window_rate_limiter_releases_expired_requests() -> None:
    limiter = SlidingWindowRateLimiter(limit=2, window_seconds=10)

    assert limiter.allow("session", now=0)
    assert limiter.allow("session", now=1)
    assert not limiter.allow("session", now=2)
    assert limiter.allow("session", now=11)
