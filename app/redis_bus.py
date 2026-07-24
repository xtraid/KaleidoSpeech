"""Redis Streams integration."""

from redis import Redis

from app.config import get_settings


def client() -> Redis:
    return Redis.from_url(get_settings().redis_url, decode_responses=False)


def publish_audio(audio: bytes) -> bytes:
    settings = get_settings()
    return client().xadd(settings.audio_stream, {"audio": audio})

