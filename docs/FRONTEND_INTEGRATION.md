# Frontend integration

## Testing con mock

1. Esegui `make frontend`.
2. Apri `http://localhost:8765/?debug=1`.
3. Premi **Start Session**.
4. Usa **Right**, **Wrong** e **Low** per simulare il backend.

I controlli mock compaiono solo con `?debug=1`. In alternativa, `make api`
serve la stessa UI da `http://127.0.0.1:8000/ui/?debug=1`.

## Testing con backend

1. Esegui `make redis`.
2. Esegui `make init` la prima volta.
3. Esegui `make build-demo` la prima volta.
4. Esegui `make dev`.
5. Apri `http://localhost:8765`.

La modalità reale usa due WebSocket con lo stesso session ID: il feed Redis
`/sessions/{session_id}/events` e l’upload PCM
`/streaming/sessions/{session_id}`. Il primo tap sul microfono apre la cattura,
converte l’audio mono in PCM s16le 16 kHz e invia frame da 40 ms; il secondo
invia `session.finish`. Per produzione migrare `ScriptProcessorNode` a un
`AudioWorklet` dopo la matrice QA browser.

Connect to `ws(s)://HOST/streaming/sessions/{session_id}`. Session IDs accept
ASCII letters, digits, `_` and `-`, up to 128 characters. When
`WEBSOCKET_AUTH_TOKEN` is configured, send `Authorization: Bearer TOKEN` (or
the `token` query parameter for clients that cannot set WebSocket headers).

Start with:

```json
{"type":"session.start","target_words":["bird"],"seed":"optional"}
```

Then send exact 40 ms frames of mono PCM signed 16-bit little-endian audio at
16 kHz (1280 bytes per frame). Each frame yields
`stream.inference.partial`. Finish with:

```json
{"type":"session.finish"}
```

The server returns one `stream.inference.final` and closes normally. Clients
must treat `RETRY` as a new capture request, `UNDECIDABLE` as insufficient
evidence, and `WRONG_WORD` as a reason code rather than a phoneme error.
`clinical_use` is always false for automated results.

Use exponential backoff with jitter for abnormal closes, but never replay audio
into a new session without explicit user consent. Close code `1009` means the
control message exceeded the configured size; `1008` indicates policy,
authentication, identifier, or rate-limit failure; `1012` requests reconnect
after server restart.

`/sessions/{session_id}/events` is the Redis event feed. Health probes are
available at `/health`, `/health/redis`, and `/health/sqlite`. Preserve the
`X-Request-ID` response header when reporting errors.
