# advX Speech Service

Python service for capturing microphone audio, publishing it through Redis
Streams, analyzing pronunciation, and storing results in SQLite.

## Requirements

- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- Docker
- PortAudio on Linux

Python does not need to be installed manually: `uv sync` downloads the required
Python version, creates `.venv`, and installs every Python dependency from
`pyproject.toml` using the exact versions in `uv.lock`.

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

## Run

```bash
uv run uvicorn app.api:app --host 127.0.0.1 --port 8000 --env-file .env --reload
```

Once the service is running locally, the FastAPI documentation is available at
`/docs`.

Capture and publish microphone frames:

```bash
uv run python -m app.audio_producer
```

The producer generates a server-side UUID and publishes to
`audio:{session_id}`. Audio is mono PCM signed 16-bit little-endian at 16 kHz,
split into 40 ms frames (640 samples and 1280 bytes).

Run the tests with:

```bash
uv run pytest
```

## Black-box cleaning contract

The cleaning implementation is intentionally not part of this repository yet.
The acceptance suite treats it as a black box:

```text
Legacy Redis record ({b"audio": mono 16 kHz float32 bytes})
    -> public cleaning function
    -> canonical mono 16 kHz float32 bytes for the solver
```

This is the current colleague-facing cleaning-test contract. The streaming
transport now follows the implementation guide and emits `pcm_s16le` int16
frames instead. The worker's future word-window aggregator must concatenate
those frames and adapt the completed window to the cleaner's public input
format. Individual 40 ms frames must never be evaluated as words.

By default the suite expects:

```python
app.cleaning.clean_audio(
    audio: bytes
) -> bytes
```

If the colleague uses another public entrypoint, select it without changing the
tests:

```bash
CLEANING_ENTRYPOINT=package.module:function uv run pytest
```

The fixtures use four real noisy/clean pairs and their official transcripts from
the public VoiceBank-DEMAND test set. The noisy tracks are converted to the
record Redis currently provides; clean tracks and transcripts remain test
oracles and are never passed to the cleaner. Tests inspect only the bytes
consumed by the solver: format, finite values, peak limit, DC offset, duration,
determinism, silence rejection, and faster-than-real-time execution. They do not
inspect or prescribe the cleaning algorithm.

## Security

Audio and personal data, especially data involving minors, must not be exposed
publicly. Production deployments require authentication, TLS, access control,
appropriate consent, and retention policies.
