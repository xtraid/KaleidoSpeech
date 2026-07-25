# Deployment

## Containers and profiles

Use `docker compose --profile development up` for a bind-mounted reload server,
or `docker compose --profile production up --build` for the non-root runtime
image. Persist both the Redis and SQLite volumes. Pre-cache the pinned phoneme
model during image construction when `PHONETIC_ENABLED=true`.

## Reverse proxy and TLS

Terminate TLS at a maintained reverse proxy or cloud load balancer. Redirect
HTTP to HTTPS, forward `X-Request-ID`, and preserve WebSocket upgrade headers:
`Upgrade`, `Connection`, `Host`, and `X-Forwarded-Proto`. Configure an idle
timeout above the maximum 15-second utterance plus processing time. Expose the
service as `wss://`; never send bearer tokens over plaintext.

Browser deployments must allow only explicit production origins at the proxy.
Do not use wildcard CORS with credentials. The current application does not
enable cross-origin HTTP access by default.

## Health and observability

Use `/health` for liveness and both `/health/sqlite` and `/health/redis` for
readiness. Scrape `/metrics`; derive p95 latency from
`advx_http_request_duration_seconds_bucket`. Alert on degraded dependencies,
open circuits, error rate, retry decisions and latency budgets.

## SQLite backup and restore

Create a consistent online backup with:

```sh
uv run python scripts/backup_sqlite.py backups/pronunciation.sqlite3
```

Verify the backup off-host and encrypt it according to the child-data policy.
To restore, stop every API instance, preserve the damaged database for
forensics, copy the verified backup to `SQLITE_PATH`, start one instance, and
check `PRAGMA integrity_check` through `/health/sqlite` before restoring
traffic. Never copy only the main file while WAL writers are active.
