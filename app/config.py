"""Application configuration loaded from environment variables."""

from functools import lru_cache
import os
from pathlib import Path

from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseModel):
    app_host: str = "127.0.0.1"
    app_port: int = Field(default=8000, ge=1, le=65535)
    sample_rate: int = Field(default=16_000, gt=0)
    frame_ms: int = Field(default=40, gt=0)
    channels: int = Field(default=1, ge=1)
    dtype: str = "int16"
    redis_url: str = "redis://localhost:6379/0"
    sqlite_path: Path = PROJECT_ROOT / "data" / "pronunciation.sqlite3"
    session_ttl_seconds: int = Field(default=900, gt=0)
    max_stream_length: int = Field(default=2_000, gt=0)

    @property
    def samples_per_frame(self) -> int:
        return self.sample_rate * self.frame_ms // 1_000

    @property
    def bytes_per_frame(self) -> int:
        return self.samples_per_frame * self.channels * 2


@lru_cache
def get_settings() -> Settings:
    """Return validated settings for the current process."""
    sqlite_path = Path(
        os.getenv(
            "SQLITE_PATH",
            os.getenv(
                "DATABASE_PATH",
                str(PROJECT_ROOT / "data" / "pronunciation.sqlite3"),
            ),
        )
    )
    if not sqlite_path.is_absolute():
        sqlite_path = PROJECT_ROOT / sqlite_path

    return Settings(
        app_host=os.getenv("APP_HOST", "127.0.0.1"),
        app_port=os.getenv("APP_PORT", "8000"),
        sample_rate=os.getenv("SAMPLE_RATE", "16000"),
        frame_ms=os.getenv("FRAME_MS", "40"),
        redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
        sqlite_path=sqlite_path,
        session_ttl_seconds=os.getenv("SESSION_TTL_SECONDS", "900"),
        max_stream_length=os.getenv("MAX_STREAM_LENGTH", "2000"),
    )


settings = get_settings()
