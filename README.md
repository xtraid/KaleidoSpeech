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
- safe ZIP ingestion for isolated English word datasets;
- deterministic speaker-disjoint train/validation/test assignment;
- versioned MFCC summary features persisted in SQLite;
- temporal MFCC trajectories with delta and delta-delta coefficients;
- speaker/channel normalization and DTW against real recording medoids;
- contrastive comparison against every active word benchmark;
- robust per-word acoustic prototypes and calibrated decision thresholds;
- explicit `CORRECT`, `INCORRECT`, `UNDECIDABLE`, and `RETRY` outcomes;
- deterministic English prompt composition for the current vocabulary;
- rolling 40 ms streaming observations with background DTW refreshes;
- sentence-benchmark streaming with live error localization over Redis;
- a runnable local worker path for WAV evaluation;
- unit, integration, black-box, and cleaning-performance tests.

Still to be integrated:

- voice activity detection and word-boundary segmentation;
- a real phonetic pronunciation engine;
- forced alignment for evaluating words inside continuous sentences;
- live Redis word-window orchestration around the runnable evaluation path;
- authentication, consent, retention, and production deployment controls;
- a client application.

The current `PronunciationEngine` remains a protocol for a future phonetic
model. The new acoustic baseline is intentionally separate: it measures
distance from observed English word examples and does not claim to recognize
phonemes.

## Architecture

```text
Isolated-word evaluation
    WAV -> cleaning -> MFCC trajectory -> DTW benchmarks -> decision
                                      \-> SQLite attempts and provenance

Live observation
    40 ms PCM frames -> FastAPI WebSocket -> rolling audio window
                                          -> background DTW observations
                                          -> provisional client events

Known-sentence benchmark
    40 ms PCM frames -> Redis Stream: audio:{session_id}
                     -> incremental alignment and error confirmation
                     -> Redis Stream: session:{session_id}:events
                     -> FastAPI WebSocket

Future continuous-speech scoring
    segmented word windows -> cleaning -> phonetic/forced-alignment engine
                                      -> SQLite final attempt
```

Redis contains ephemeral audio, session state, and UI events. SQLite contains
durable benchmark definitions, feature provenance, calibrated thresholds, and
final pronunciation attempts. Streaming distance events are observations rather
than clinical or educational diagnoses.

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
  -d redis:8.8.0-alpine \
  redis-server --appendonly yes
```

After the first run, restart the existing container with:

```bash
docker start advx-redis
```

Verify that Redis 8 exposes both transport and vector-search capabilities:

```bash
uv run python -m scripts.check_redis
```

Populate the optional HNSW index from accepted SQLite MFCC summaries:

```bash
SQLITE_PATH=/tmp/advx-demo.sqlite3 \
  uv run python -m scripts.sync_vector_index \
  --query-recording-id 1 -k 5
```

The synchronization is idempotent. It computes one global normalization for
the selected locale, upserts keys by immutable recording ID, and stores the
normalization version alongside the index.

Redis is optional for local WAV evaluation. SQLite remains authoritative; Redis
8 provides streams, cache, Search/HNSW retrieval, and the native Vector Set
commands. A core-only Redis/Valkey package can run streams but cannot enable the
optional `FT.SEARCH` index.

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
- `GET /health/redis`
- `WS /sessions/{session_id}/events`
- `WS /streaming/sessions/{session_id}`
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

## Build and try the English acoustic baseline

The `DS` branch includes three sample archives with five isolated words:
`bird`, `follow`, `forward`, `happy`, and `learn`. Build a quick benchmark with
100 recordings per word:

```bash
uv run python -m scripts.build_acoustic_benchmarks \
  --limit-per-word 100 \
  tmp/1.zip tmp/2.zip tmp/3.zip
```

Evaluate a mono 16 kHz WAV:

```bash
uv run python -m app.worker evaluate-wav bird path/to/bird.wav
```

The import reads archive members without extracting them, ignores macOS
metadata, deduplicates identical audio, pseudonymizes speaker identifiers, and
keeps each speaker in exactly one global split. SQLite stores derived features,
prototypes, thresholds, provenance versions, and decisions; it does not store
raw WAV bytes.

The default worker uses the temporal DTW engine. Pass `--engine summary` only
to compare against the original fixed-vector baseline.

The five-word fragment contains only positive examples of isolated commands.
Other words provide provisional negatives, not labelled pronunciation errors.
Its thresholds are therefore suitable for pipeline experiments, not production
or educational claims. Sentence prompts can be generated now, but evaluating
their natural continuous speech requires a forced aligner and separately
calibrated continuous-speech benchmarks.

The DTW prototype is deliberately conservative and exposes its closest
competitor and margin. Adversarial local tests show that it reduces cross-word
false acceptance, but it does not solve demographic coverage or phonetic error
detection by itself. A production scorer still needs balanced speaker data and
phone-labelled learner speech.

Convenience targets are available:

```bash
make init
make test
make redis
make api
make build-demo
```

## Audio and cleaning contract

Every Redis audio record contains:

```text
sequence, captured_at_ns, sample_rate, frame_ms, pcm_s16le
```

Each record represents one 40 ms frame: 640 mono samples and 1280 bytes at
16 kHz. Cleaning and pronunciation evaluation operate on a complete word window
assembled from ordered, contiguous records; individual frames are never treated
as words.

The public PCM cleaning boundary is:

```python
app.cleaning.clean_audio(pcm_s16le: bytes) -> bytes
```

Its explicit name is `app.cleaning.clean_pcm`; `clean_audio` remains as a
backward-compatible alias. The file-based implementation is named
`app.audio_cleaning.clean_wav_files`, avoiding ambiguity between PCM and path
contracts.

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

## Demo streaming microfono (40 ms)

Il prototipo propone una frase, registra PCM mono 16 kHz in blocchi esatti da
40 ms e stampa un evento JSON per blocco. VAD e qualità sono aggiornati a ogni
frame; il confronto DTW viene aggiornato ogni 200 ms su una finestra mobile.

Per ricevere soltanto un responso provvisorio ogni 40 ms su una singola parola,
usa la modalità focalizzata:

```bash
uv run python -m scripts.streaming_demo bird \
  --duration 8 \
  --decision-only
```

Nei primi frame e durante il silenzio stampa `ATTENDI`; appena la finestra
contiene evidenza sufficiente stampa `CORRETTO`, `NON CORRETTO` oppure
`INCERTO`. Il giudizio riguarda la finestra audio mobile, non un singolo frame
isolato.

```bash
SQLITE_PATH=/tmp/advx-demo.sqlite3 \
  uv run python -m scripts.streaming_demo \
  bird follow forward happy learn --duration 8
```

Il WebSocket `/streaming/sessions/{session_id}` accetta prima
`{"type":"session.start","target_words":["bird","happy"]}`, poi frame binari
PCM s16le da 1.280 byte, infine `{"type":"session.finish"}`. Una frase termina
con `REVIEW_REQUIRED`: senza forced alignment pediatrico validato non viene
inventato un verdetto parola-per-parola.

### Frase benchmark con localizzazione live dell'errore

Il percorso MVP per una frase nota usa Redis come trasporto effettivo dei frame:
la demo pubblica ogni blocco da 40 ms su `audio:{session_id}`, rilegge il record,
aggiorna l'allineamento DTW e pubblica gli eventi su
`session:{session_id}:events`.

Inizializza prima le nuove tabelle:

```bash
uv run python -m scripts.init_db
```

Prepara un JSON con i confini delle parole del WAV benchmark:

```json
[
  {"text": "the", "start_ms": 0, "end_ms": 180},
  {"text": "happy", "start_ms": 180, "end_ms": 520},
  {"text": "bird", "start_ms": 520, "end_ms": 850}
]
```

Registra il benchmark (WAV mono 16 kHz):

```bash
uv run python -m scripts.build_sentence_benchmark \
  en-001 "The happy bird" benchmark.wav units.json
```

Avvia Redis e prova lo streaming:

```bash
uv run python -m scripts.sentence_streaming_demo en-001 --duration 8
```

L'evento `pronunciation.error` contiene sia `start_ms`, cioè l'inizio stimato
del segmento problematico, sia `detected_at_ms`, cioè quando l'evidenza è
diventata sufficiente. La conferma richiede due blocchi acustici da 200 ms:
l'aggiornamento audio resta di 40 ms, mentre la latenza nominale del giudizio è
circa 400 ms. Le soglie locali e finali appartengono al benchmark e andranno
calibrate sul dataset, non sul singolo parlante di riferimento.

## Security

Audio and personal data, especially data involving minors, must not be exposed
publicly. The current service binds locally by default and is not
production-ready. A production deployment requires authentication, TLS, access
control, appropriate consent, retention policies, and secrets management.
