from app.api import health, metrics, redis_health, sqlite_health
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


def test_sqlite_health_checks_integrity(tmp_path, monkeypatch) -> None:
    from app.config import Settings

    monkeypatch.setattr(
        "app.database.get_settings",
        lambda: Settings(sqlite_path=tmp_path / "health.sqlite3"),
    )
    assert sqlite_health() == {"status": "ok", "sqlite": "ok"}


def test_metrics_uses_prometheus_text_format() -> None:
    payload = metrics()
    assert "# TYPE advx_http_requests_total counter" in payload
    assert "# TYPE advx_decisions_total counter" in payload


def test_frontend_redirect_targets_static_mount() -> None:
    from app.api import FRONTEND_DIR, frontend_redirect

    assert (FRONTEND_DIR / "index.html").is_file()
    assert frontend_redirect().headers["location"] == "/ui/"
