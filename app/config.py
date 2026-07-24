"""Application configuration loaded from environment variables."""

from functools import lru_cache
import os
from pathlib import Path

from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseModel):
    app_host: str = "127.0.0.1"
    app_port: int = Field(default=8000, ge=1, le=65535)
    database_path: Path = PROJECT_ROOT / "data" / "pronunciation.sqlite3"
    redis_url: str = "redis://localhost:6379/0"
    audio_stream: str = "pronunciation:audio"


@lru_cache
def get_settings() -> Settings:
    """Return validated settings for the current process."""
    database_path = Path(
        os.getenv(
            "DATABASE_PATH",
            str(PROJECT_ROOT / "data" / "pronunciation.sqlite3"),
        )
    )
    if not database_path.is_absolute():
        database_path = PROJECT_ROOT / database_path

    return Settings(
        app_host=os.getenv("APP_HOST", "127.0.0.1"),
        app_port=os.getenv("APP_PORT", "8000"),
        database_path=database_path,
        redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
        audio_stream=os.getenv("AUDIO_STREAM", "pronunciation:audio"),
    )

