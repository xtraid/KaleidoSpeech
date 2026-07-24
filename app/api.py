"""HTTP and WebSocket API for the speech service."""

import asyncio
import json

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from app.redis_bus import client


app = FastAPI(title="Pronunciation Speech Service")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.websocket("/sessions/{session_id}/events")
async def session_events(websocket: WebSocket, session_id: str) -> None:
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
