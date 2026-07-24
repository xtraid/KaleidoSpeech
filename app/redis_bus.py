"""Redis Streams, cache, and ephemeral session-state integration."""

import json
from typing import Any

from redis import Redis

from app.config import get_settings


client: Redis = Redis.from_url(
    get_settings().redis_url,
    decode_responses=False,
    socket_timeout=2,
    health_check_interval=30,
)


def publish_audio_frame(
    session_id: str,
    sequence: int,
    captured_at_ns: int,
    pcm: bytes,
) -> bytes:
    settings = get_settings()
    if len(pcm) != settings.bytes_per_frame:
        raise ValueError(
            f"Expected {settings.bytes_per_frame} PCM bytes, got {len(pcm)}"
        )

    return client.xadd(
        f"audio:{session_id}",
        {
            b"sequence": str(sequence).encode(),
            b"captured_at_ns": str(captured_at_ns).encode(),
            b"sample_rate": str(settings.sample_rate).encode(),
            b"frame_ms": str(settings.frame_ms).encode(),
            b"pcm_s16le": pcm,
        },
        maxlen=settings.max_stream_length,
        approximate=True,
    )


def update_session_state(session_id: str, state: dict[str, Any]) -> None:
    settings = get_settings()
    key = f"session:{session_id}:state"
    encoded = {
        str(field): json.dumps(value, ensure_ascii=False)
        for field, value in state.items()
    }
    pipeline = client.pipeline()
    pipeline.hset(key, mapping=encoded)
    pipeline.expire(key, settings.session_ttl_seconds)
    pipeline.execute()


def publish_ui_event(session_id: str, event: dict[str, Any]) -> bytes:
    return client.xadd(
        f"session:{session_id}:events",
        {b"json": json.dumps(event, ensure_ascii=False).encode()},
        maxlen=500,
        approximate=True,
    )
