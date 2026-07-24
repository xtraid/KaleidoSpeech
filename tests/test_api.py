from app.api import health


def test_health() -> None:
    assert health() == {"status": "ok"}
