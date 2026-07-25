"""Redis Streams, cache, and ephemeral session-state integration."""

from dataclasses import asdict, dataclass
import json
from typing import Any

from redis import Redis
from redis.exceptions import RedisError

from app.config import get_settings


client: Redis = Redis.from_url(
    get_settings().redis_url,
    decode_responses=False,
    socket_timeout=2,
    health_check_interval=30,
)


@dataclass(frozen=True)
class RedisCapabilities:
    reachable: bool
    version: str | None
    search: bool
    vector_set: bool
    error: str | None = None

    def as_dict(self) -> dict[str, bool | str | None]:
        return asdict(self)


def _command_available(redis_client: Redis, command: str) -> bool:
    response = redis_client.execute_command("COMMAND", "INFO", command)
    if response is None:
        return False
    if isinstance(response, (list, tuple)):
        return bool(response) and response[0] is not None
    return bool(response)


def probe_capabilities(
    *, redis_client: Redis | None = None,
) -> RedisCapabilities:
    """Inspect Redis without changing state; failures are returned as data."""
    runtime = redis_client or client
    try:
        runtime.ping()
        server = runtime.info(section="server")
        version = server.get("redis_version")
        return RedisCapabilities(
            reachable=True,
            version=str(version) if version is not None else None,
            search=_command_available(runtime, "FT.SEARCH"),
            vector_set=_command_available(runtime, "VSIM"),
        )
    except (RedisError, OSError, RuntimeError) as error:
        return RedisCapabilities(
            reachable=False,
            version=None,
            search=False,
            vector_set=False,
            error=f"{type(error).__name__}: {error}",
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
