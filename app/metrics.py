"""Minimal Prometheus text exposition for service-level metrics."""

from __future__ import annotations

from collections import Counter
from threading import Lock


_lock = Lock()
_requests: Counter[tuple[str, str, int]] = Counter()
_request_duration_seconds: Counter[tuple[str, str]] = Counter()
_request_duration_buckets: Counter[tuple[str, str, float]] = Counter()
_decisions: Counter[str] = Counter()
_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)


def observe_request(method: str, path: str, status: int, duration: float) -> None:
    with _lock:
        _requests[(method, path, status)] += 1
        _request_duration_seconds[(method, path)] += duration
        for bucket in _BUCKETS:
            if duration <= bucket:
                _request_duration_buckets[(method, path, bucket)] += 1


def observe_decision(status: str) -> None:
    with _lock:
        _decisions[status] += 1


def exposition() -> str:
    lines = [
        "# HELP advx_http_requests_total HTTP requests processed.",
        "# TYPE advx_http_requests_total counter",
    ]
    with _lock:
        for (method, path, status), count in sorted(_requests.items()):
            lines.append(
                f'advx_http_requests_total{{method="{method}",path="{path}",'
                f'status="{status}"}} {count}'
            )
        lines.extend(
            [
                "# HELP advx_http_request_duration_seconds_total "
                "Cumulative HTTP request duration.",
                "# TYPE advx_http_request_duration_seconds_total counter",
            ]
        )
        for (method, path), duration in sorted(_request_duration_seconds.items()):
            lines.append(
                "advx_http_request_duration_seconds_total"
                f'{{method="{method}",path="{path}"}} {duration:.9f}'
            )
        lines.extend(
            [
                "# HELP advx_http_request_duration_seconds "
                "HTTP request latency histogram.",
                "# TYPE advx_http_request_duration_seconds histogram",
            ]
        )
        request_counts: Counter[tuple[str, str]] = Counter()
        for (method, path, _), count in _requests.items():
            request_counts[(method, path)] += count
        for (method, path, bucket), count in sorted(
            _request_duration_buckets.items()
        ):
            lines.append(
                "advx_http_request_duration_seconds_bucket"
                f'{{method="{method}",path="{path}",le="{bucket}"}} {count}'
            )
        for (method, path), count in sorted(request_counts.items()):
            lines.append(
                "advx_http_request_duration_seconds_bucket"
                f'{{method="{method}",path="{path}",le="+Inf"}} {count}'
            )
        lines.extend(
            [
                "# HELP advx_decisions_total Final decisions by status.",
                "# TYPE advx_decisions_total counter",
            ]
        )
        for status, count in sorted(_decisions.items()):
            lines.append(f'advx_decisions_total{{status="{status}"}} {count}')
    return "\n".join(lines) + "\n"
