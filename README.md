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

API: <http://127.0.0.1:8000> · Docs: <http://127.0.0.1:8000/docs>

Run the tests with:

```bash
uv run pytest
```

## Security

Audio and personal data, especially data involving minors, must not be exposed
publicly. Production deployments require authentication, TLS, access control,
appropriate consent, and retention policies.
