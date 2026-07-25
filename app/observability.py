"""Small dependency-free observability primitives for the API."""

from __future__ import annotations

from contextvars import ContextVar
import json
import logging
import time
from uuid import uuid4

from fastapi import Request, Response

from app.config import get_settings
from app.metrics import observe_request


correlation_id: ContextVar[str] = ContextVar("correlation_id", default="-")


class JsonFormatter(logging.Formatter):
    """Serialize standard log records without leaking arbitrary object state."""

    def format(self, record: logging.LogRecord) -> str:
        return json.dumps(
            {
                "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
                "correlation_id": correlation_id.get(),
            },
            ensure_ascii=False,
        )


def configure_logging() -> None:
    settings = get_settings()
    handler = logging.StreamHandler()
    if settings.log_json:
        handler.setFormatter(JsonFormatter())
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        handlers=[handler],
        force=True,
    )


async def correlation_middleware(request: Request, call_next) -> Response:
    request_id = request.headers.get("x-request-id") or str(uuid4())
    token = correlation_id.set(request_id[:128])
    started = time.monotonic()
    try:
        response = await call_next(request)
        response.headers["x-request-id"] = correlation_id.get()
        duration = time.monotonic() - started
        observe_request(request.method, request.url.path, response.status_code, duration)
        logging.getLogger("advx.request").info(
            "%s %s %s %.2fms",
            request.method,
            request.url.path,
            response.status_code,
            duration * 1_000,
        )
        return response
    finally:
        correlation_id.reset(token)
