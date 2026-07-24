from app.api import health, redis_health
from app.redis_bus import RedisCapabilities


def test_health() -> None:
    assert health() == {"status": "ok"}


def test_redis_health_exposes_capabilities(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.api.probe_capabilities",
        lambda: RedisCapabilities(
            reachable=True,
            version="8.8.0",
            search=True,
            vector_set=True,
        ),
    )
    assert redis_health() == {
        "status": "ok",
        "redis": {
            "reachable": True,
            "version": "8.8.0",
            "search": True,
            "vector_set": True,
            "error": None,
        },
    }
