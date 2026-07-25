"""Authentication, validation and in-process rate limiting helpers."""

from __future__ import annotations

from collections import defaultdict, deque
import hmac
import time

from fastapi import WebSocket

from app.config import Settings


class SlidingWindowRateLimiter:
    def __init__(self, limit: int, window_seconds: int) -> None:
        self.limit = limit
        self.window_seconds = window_seconds
        self._requests: dict[str, deque[float]] = defaultdict(deque)

    def allow(self, key: str, *, now: float | None = None) -> bool:
        current = time.monotonic() if now is None else now
        requests = self._requests[key]
        cutoff = current - self.window_seconds
        while requests and requests[0] <= cutoff:
            requests.popleft()
        if len(requests) >= self.limit:
            return False
        requests.append(current)
        return True


def websocket_authorized(websocket: WebSocket, settings: Settings) -> bool:
    """Optional bearer/query-token stub; disabled when no token is configured."""
    expected = settings.websocket_auth_token
    if expected is None:
        return True
    authorization = websocket.headers.get("authorization", "")
    supplied = (
        authorization.removeprefix("Bearer ").strip()
        if authorization.startswith("Bearer ")
        else websocket.query_params.get("token", "")
    )
    return hmac.compare_digest(supplied, expected)
