"""HTTP and WebSocket API for the speech service."""

import asyncio
import json
import re

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from app.redis_bus import client, probe_capabilities
from app.streaming_inference import StreamingInferenceSession


app = FastAPI(title="Pronunciation Speech Service")
SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/health/redis")
def redis_health() -> dict:
    capabilities = probe_capabilities()
    state = "ok" if capabilities.reachable else "degraded"
    return {"status": state, "redis": capabilities.as_dict()}


@app.websocket("/sessions/{session_id}/events")
async def session_events(websocket: WebSocket, session_id: str) -> None:
    if SESSION_ID_PATTERN.fullmatch(session_id) is None:
        await websocket.close(code=1008, reason="Invalid session_id")
        return
    await websocket.accept()
    stream = f"session:{session_id}:events"
    last_id: bytes | str = "$"

    try:
        while True:
            batches = await asyncio.to_thread(
                client.xread,
                {stream: last_id},
                count=10,
                block=1_000,
            )
            for _, messages in batches:
                for message_id, fields in messages:
                    last_id = message_id
                    await websocket.send_json(json.loads(fields[b"json"]))
    except WebSocketDisconnect:
        return


@app.websocket("/streaming/sessions/{session_id}")
async def streaming_audio(websocket: WebSocket, session_id: str) -> None:
    """Accept a start message, then exact 40 ms binary PCM frames."""
    if SESSION_ID_PATTERN.fullmatch(session_id) is None:
        await websocket.close(code=1008, reason="Invalid session_id")
        return
    await websocket.accept()
    inference: StreamingInferenceSession | None = None
    try:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                return
            if message.get("text") is not None:
                try:
                    command = json.loads(message["text"])
                    command_type = command.get("type")
                    if command_type == "session.start" and inference is None:
                        targets = command.get("target_words")
                        if not isinstance(targets, list) or not targets:
                            raise ValueError("target_words must be a non-empty list")
                        inference = await asyncio.to_thread(
                            StreamingInferenceSession,
                            targets,
                            seed=command.get("seed"),
                        )
                        await websocket.send_json(inference.start_event())
                    elif command_type == "session.finish" and inference is not None:
                        await websocket.send_json(inference.finish())
                        await websocket.close(code=1000)
                        return
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
