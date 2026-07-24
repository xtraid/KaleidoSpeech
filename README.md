# advX Speech Service

Servizio Python per acquisire audio dal microfono, distribuirlo tramite Redis
Streams, analizzare la pronuncia e salvare i risultati in SQLite.

## Requisiti

- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- Docker
- PortAudio

Il progetto usa Python 3.11+, Redis 7+ e SQLite, già incluso in Python.

Su Arch/CachyOS, PortAudio si installa con:

```bash
sudo pacman -S portaudio
```

## Configurazione

Dalla root del repository:

```bash
uv sync
cp .env.example .env
uv run python -m scripts.init_db
```

`uv sync` installa automaticamente Python, crea `.venv` e sincronizza le
dipendenze definite in `pyproject.toml` e `uv.lock`.

Avvia Redis:

```bash
docker run --name advx-redis \
  -p 127.0.0.1:6379:6379 \
  -d redis:7-alpine \
  redis-server --appendonly yes
```

## Avvio

```bash
uv run uvicorn app.api:app --host 127.0.0.1 --port 8000 --env-file .env --reload
```

API: <http://127.0.0.1:8000> · Documentazione: <http://127.0.0.1:8000/docs>

## Test

```bash
uv run pytest
```

## Sicurezza

Audio e dati personali, soprattutto se relativi a minori, non devono essere
esposti pubblicamente. In produzione sono necessari autenticazione, TLS,
controllo degli accessi, consenso e policy di retention adeguate.
