# advX Speech Service

Backend prototype for a pronunciation-training application. The service captures
microphone audio, publishes canonical PCM frames through Redis Streams, cleans
complete word windows, stores pronunciation benchmarks and attempts in SQLite,
and exposes session events through a FastAPI WebSocket.

## Project status

This project is currently an early-stage prototype.

Implemented:

- microphone capture as mono 16 kHz signed 16-bit little-endian PCM;
- 40 ms audio frames published to Redis Streams;
- validation and aggregation of contiguous frames into complete word windows;
- deterministic audio-cleaning pipeline;
- SQLite schema and repositories for benchmarks and pronunciation attempts;
- Redis-backed session state, benchmark cache, and UI event streams;
- FastAPI health endpoint and WebSocket event forwarding;
- unit, integration, black-box, and cleaning-performance tests.

Still to be integrated:

- voice activity detection and word-boundary segmentation;
- a real phonetic pronunciation engine;
- the end-to-end worker that connects aggregation, cleaning, evaluation, and
  persistence;
- authentication, consent, retention, and production deployment controls;
- a client application.

The current `PronunciationEngine` is a protocol (an interface contract), not an
implemented speech-recognition model. Likewise, `app.worker` provides reusable
consumer and persistence primitives but is not yet a standalone end-to-end
worker.

## Architecture

```text
Microphone producer
    -> Redis Stream: audio:{session_id}
    -> validation and contiguous word-window aggregation
    -> audio cleaning
    -> pronunciation engine (to be integrated)
    -> SQLite final attempt
    -> Redis Stream: session:{session_id}:events
    -> FastAPI WebSocket
```

Redis contains ephemeral audio, session state, and UI events. SQLite contains
durable benchmark definitions and final pronunciation attempts.

## Requirements

- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- Docker
- PortAudio on Linux

Python does not need to be installed manually: `uv sync` downloads the required
Python version, creates `.venv`, and installs the dependencies declared in
`pyproject.toml` at the versions resolved in `uv.lock`.

### CachyOS / Arch Linux

```bash
sudo pacman -S uv portaudio docker
sudo systemctl enable --now docker
```

### macOS

Install [Docker Desktop](https://docs.docker.com/desktop/setup/install/mac-install/)
and uv:

```bash
brew install uv
```

The `sounddevice` package installed by `uv sync` includes PortAudio on macOS.

## Setup

From the repository root:

```bash
uv sync
cp .env.example .env
uv run python -m scripts.init_db
```

The initialization command creates the SQLite schema and seeds the initial
Italian benchmark for `cane`. It is safe to run more than once.

Start Redis:

```bash
docker run --name advx-redis \
  -p 127.0.0.1:6379:6379 \
  -d redis:7-alpine \
  redis-server --appendonly yes
```

After the first run, restart the existing container with:

```bash
docker start advx-redis
```

## Run the available components

Start the API:

```bash
uv run uvicorn app.api:app \
  --host 127.0.0.1 \
  --port 8000 \
  --env-file .env \
  --reload
```

Available endpoints:

- `GET /health`
- `WS /sessions/{session_id}/events`
- interactive FastAPI documentation at `/docs`

Capture and publish microphone frames:

```bash
uv run python -m app.audio_producer
```

The producer generates a server-side UUID and prints it to the terminal. Press
Enter to stop capturing. Frames are published to `audio:{session_id}`.

Running these two processes does not yet produce a pronunciation score: the
voice-activity/word segmentation and pronunciation-engine integration are still
pending.

## Audio and cleaning contract

Every Redis audio record contains:

```text
sequence, captured_at_ns, sample_rate, frame_ms, pcm_s16le
```

Each record represents one 40 ms frame: 640 mono samples and 1280 bytes at
16 kHz. Cleaning and pronunciation evaluation operate on a complete word window
assembled from ordered, contiguous records; individual frames are never treated
as words.

The public cleaning boundary is:

```python
app.cleaning.clean_audio(pcm_s16le: bytes) -> bytes
```

Both input and output are canonical mono 16 kHz signed int16 little-endian PCM.
The implementation may use normalized float32 samples internally, but float32
is not part of the public boundary.

The black-box suite can test a compatible alternative implementation without
changing the tests:

```bash
CLEANING_ENTRYPOINT=package.module:function uv run pytest
```

Cleaning fixtures contain four public noisy/clean VoiceBank-DEMAND pairs and
their official transcripts. The clean tracks and transcripts are test oracles;
they are never passed to the cleaner.

## Tests

Run the complete suite:

```bash
uv run pytest
```

The suite covers API health, audio framing, Redis adapter behavior, SQLite
transactions and input handling, cleaning quality and performance, contiguous
word-window aggregation, pronunciation-engine invocation, and final attempt
persistence. Most tests use isolated databases and Redis doubles; tests that
need a live Redis instance are skipped when it is unavailable.

## Security

Audio and personal data, especially data involving minors, must not be exposed
publicly. The current service binds locally by default and is not
production-ready. A production deployment requires authentication, TLS, access
control, appropriate consent, retention policies, and secrets management.
