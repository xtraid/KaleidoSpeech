"""HTTP and WebSocket API for the speech service."""

import asyncio
from contextlib import asynccontextmanager
import json
import logging
import re
import sqlite3
from uuid import uuid4
import hmac
from pathlib import Path

from fastapi import (
    FastAPI,
    Depends,
    Header,
    HTTPException,
    Request,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import JSONResponse
from fastapi.responses import PlainTextResponse
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from redis.exceptions import RedisError

from app.config import get_settings
from app.database import db_connection
from app.decision_repository import (
    annotate_decision,
    list_unreviewed_decisions,
    save_decision_log,
)
from app.frontend_contract import pronunciation_evaluated_event
from app.ensemble_inference import get_ensemble_decision_engine
from app.metrics import exposition, observe_decision
from app.observability import configure_logging, correlation_middleware
from app.phonetic_runtime import (
    get_final_evaluator,
    get_phoneme_model,
    get_sentence_final_evaluator,
)
from app.redis_bus import client, probe_capabilities
from app.redis_bus import publish_ui_event
from app.resilience import CircuitBreaker, CircuitOpenError, call_with_retry
from app.security import SlidingWindowRateLimiter, websocket_authorized
from app.streaming_inference import StreamingInferenceSession


SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
logger = logging.getLogger(__name__)
active_websockets: set[WebSocket] = set()


@asynccontextmanager
async def lifespan(_: FastAPI):
    configure_logging()
    await asyncio.to_thread(get_ensemble_decision_engine)
    if get_settings().phonetic_enabled:
        # Pay the model load/optional quantization cost before the first
        # microphone attempt instead of blocking session.start.
        await asyncio.to_thread(get_phoneme_model)
    yield
    sockets = tuple(active_websockets)
    if sockets:
        await asyncio.gather(
            *(
                socket.close(code=1012, reason="Server shutting down")
                for socket in sockets
            ),
            return_exceptions=True,
        )


app = FastAPI(title="Pronunciation Speech Service", lifespan=lifespan)
app.middleware("http")(correlation_middleware)
settings = get_settings()
http_rate_limiter = SlidingWindowRateLimiter(
    settings.api_rate_limit_requests,
    settings.api_rate_limit_window_seconds,
)
websocket_rate_limiter = SlidingWindowRateLimiter(
    settings.api_rate_limit_requests,
    settings.api_rate_limit_window_seconds,
)
redis_circuit_breaker = CircuitBreaker()


class DecisionAnnotation(BaseModel):
    ground_truth: str = Field(pattern="^(CORRECT|INCORRECT|RETRY|UNDECIDABLE)$")
    reviewer_id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")


def require_admin_token(x_admin_token: str | None = Header(default=None)) -> None:
    expected = get_settings().admin_api_token
    if expected is None:
        raise HTTPException(status_code=503, detail="Review API is not configured")
    if x_admin_token is None or not hmac.compare_digest(x_admin_token, expected):
        raise HTTPException(status_code=401, detail="Unauthorized")


@app.middleware("http")
async def rate_limit_http(request: Request, call_next):
    client_host = request.client.host if request.client else "unknown"
    if not http_rate_limiter.allow(client_host):
        return JSONResponse(
            {"detail": "Rate limit exceeded"},
            status_code=429,
            headers={"Retry-After": str(settings.api_rate_limit_window_seconds)},
        )
    return await call_next(request)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/health/redis")
def redis_health() -> dict:
    capabilities = probe_capabilities()
    state = "ok" if capabilities.reachable else "degraded"
    return {"status": state, "redis": capabilities.as_dict()}


@app.get("/health/sqlite")
def sqlite_health() -> dict[str, str]:
    try:
        with db_connection() as database:
            database.execute("SELECT 1").fetchone()
            integrity = database.execute("PRAGMA quick_check").fetchone()[0]
        state = "ok" if integrity == "ok" else "degraded"
        return {"status": state, "sqlite": integrity}
    except sqlite3.Error as error:
        logger.warning("SQLite health check failed: %s", error)
        return {"status": "degraded", "sqlite": type(error).__name__}


@app.get("/metrics", response_class=PlainTextResponse)
def metrics() -> str:
    return exposition()


@app.get("/review/decisions")
def review_queue(
    limit: int = 50,
    _: None = Depends(require_admin_token),
) -> list[dict]:
    return list_unreviewed_decisions(limit=limit)


@app.post("/review/decisions/{attempt_id}")
def review_decision(
    attempt_id: str,
    annotation: DecisionAnnotation,
    _: None = Depends(require_admin_token),
) -> dict[str, bool]:
    if SESSION_ID_PATTERN.fullmatch(attempt_id) is None:
        raise HTTPException(status_code=422, detail="Invalid attempt_id")
    updated = annotate_decision(
        attempt_id,
        ground_truth=annotation.ground_truth,
        reviewer_id=annotation.reviewer_id,
    )
    if not updated:
        raise HTTPException(status_code=404, detail="Attempt not found")
    return {"updated": True}


async def _authorize_websocket(websocket: WebSocket, session_id: str) -> bool:
    settings = get_settings()
    if SESSION_ID_PATTERN.fullmatch(session_id) is None:
        await websocket.close(code=1008, reason="Invalid session_id")
        return False
    client_host = websocket.client.host if websocket.client else "unknown"
    if not websocket_rate_limiter.allow(f"{client_host}:{session_id}"):
        await websocket.close(code=1008, reason="Rate limit exceeded")
        return False
    if not websocket_authorized(websocket, settings):
        await websocket.close(code=1008, reason="Unauthorized")
        return False
    return True


@app.websocket("/sessions/{session_id}/events")
async def session_events(websocket: WebSocket, session_id: str) -> None:
    if not await _authorize_websocket(websocket, session_id):
        return
    await websocket.accept()
    active_websockets.add(websocket)
    stream = f"session:{session_id}:events"
    last_id: bytes | str = "$"

    try:
        while True:
            batches = await asyncio.to_thread(
                call_with_retry,
                client.xread,
                {stream: last_id},
                count=10,
                block=1_000,
                breaker=redis_circuit_breaker,
                retry_on=(RedisError, OSError, RuntimeError),
            )
            for _, messages in batches:
                for message_id, fields in messages:
                    last_id = message_id
                    await websocket.send_json(json.loads(fields[b"json"]))
    except WebSocketDisconnect:
        return
    finally:
        active_websockets.discard(websocket)


FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
if FRONTEND_DIR.is_dir():
    app.mount("/ui", StaticFiles(directory=FRONTEND_DIR, html=True), name="ui")

    @app.get("/", include_in_schema=False)
    def frontend_redirect() -> RedirectResponse:
        return RedirectResponse("/ui/")


@app.websocket("/streaming/sessions/{session_id}")
async def streaming_audio(websocket: WebSocket, session_id: str) -> None:
    """Accept a start message, then exact 40 ms binary PCM frames."""
    if not await _authorize_websocket(websocket, session_id):
        return
    await websocket.accept()
    active_websockets.add(websocket)
    inference: StreamingInferenceSession | None = None
    try:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                return
            if message.get("text") is not None:
                try:
                    if (
                        len(message["text"].encode("utf-8"))
                        > get_settings().max_websocket_message_bytes
                    ):
                        await websocket.close(code=1009, reason="Message too large")
                        return
                    command = json.loads(message["text"])
                    if not isinstance(command, dict):
                        raise ValueError("Command must be a JSON object")
                    command_type = command.get("type")
                    if command_type == "session.start" and inference is None:
                        targets = command.get("target_words")
                        if not isinstance(targets, list) or not targets:
                            raise ValueError("target_words must be a non-empty list")
                        if (
                            len(targets) > 20
                            or any(
                                not isinstance(word, str)
                                or not word.strip()
                                or len(word) > 64
                                for word in targets
                            )
                        ):
                            raise ValueError("target_words contains invalid values")
                        expected_text = command.get("expected_text")
                        if expected_text is not None and (
                            not isinstance(expected_text, str)
                            or not expected_text.strip()
                            or len(expected_text) > 500
                        ):
                            raise ValueError("expected_text must be a non-empty string")
                        final_evaluator = (
                            await asyncio.to_thread(get_final_evaluator)
                            if get_settings().phonetic_enabled
                            and expected_text is None
                            else None
                        )
                        sentence_final_evaluator = (
                            await asyncio.to_thread(get_sentence_final_evaluator)
                            if get_settings().phonetic_enabled
                            and expected_text is not None
                            else None
                        )
                        inference = await asyncio.to_thread(
                            StreamingInferenceSession,
                            targets,
                            seed=command.get("seed"),
                            final_evaluator=final_evaluator,
                            sentence_final_evaluator=sentence_final_evaluator,
                            locale=command.get("locale", "en"),
                            expected_text=expected_text,
                        )
                        await websocket.send_json(inference.start_event())
                    elif command_type == "session.finish" and inference is not None:
                        result = inference.finish()
                        observe_decision(str(result.get("status", "UNKNOWN")))
                        attempt_id = f"attempt_{uuid4().hex}"
                        try:
                            await asyncio.to_thread(
                                save_decision_log,
                                attempt_id=attempt_id,
                                session_id=session_id,
                                result=result,
                            )
                        except (sqlite3.Error, TypeError, ValueError) as error:
                            logger.warning("Decision audit persistence failed: %s", error)
                        evaluated_event = pronunciation_evaluated_event(
                            attempt_id=attempt_id,
                            session_id=session_id,
                            result=result,
                        )
                        # Return directly before Redis fan-out so a delayed or
                        # unavailable event bus cannot hold the browser UI.
                        await websocket.send_json(evaluated_event)
                        try:
                            await asyncio.to_thread(
                                call_with_retry,
                                publish_ui_event,
                                session_id,
                                evaluated_event,
                                breaker=redis_circuit_breaker,
                                retry_on=(RedisError, OSError, RuntimeError),
                            )
                        except (
                            RedisError,
                            OSError,
                            RuntimeError,
                            CircuitOpenError,
                        ) as error:
                            logger.warning("Redis event publication failed: %s", error)
                        await websocket.close(code=1000)
                        return
                    elif command_type == "ping":
                        await websocket.send_json({"type": "pong"})
                    else:
                        raise ValueError("Invalid command or session state")
                except (json.JSONDecodeError, TypeError, ValueError, LookupError) as error:
                    await websocket.send_json(
                        {"type": "stream.error", "message": str(error)}
                    )
            elif message.get("bytes") is not None:
                if inference is None:
                    await websocket.send_json(
                        {"type": "stream.error",
                         "message": "Send session.start before audio"}
                    )
                    continue
                try:
                    event = await asyncio.to_thread(
                        inference.push_frame, message["bytes"]
                    )
                    await websocket.send_json(event)
                except (ValueError, RuntimeError) as error:
                    await websocket.send_json(
                        {"type": "stream.error", "message": str(error)}
                    )
    except WebSocketDisconnect:
        return
    finally:
        active_websockets.discard(websocket)
