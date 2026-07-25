# advX Speech Service

Guida rapida per la demo a frasi:
[`docs/HACKATHON_DEMO.md`](docs/HACKATHON_DEMO.md).

Backend prototype for a pronunciation-training application. The service captures
microphone audio, publishes canonical PCM frames through Redis Streams, scores
complete sentence utterances with a phonetic backend, streams stable per-word
updates while the learner is still speaking, stores pronunciation benchmarks and
attempts in SQLite, and exposes session events through a FastAPI WebSocket.

## Vedere subito il prodotto

Per vedere l'interfaccia senza Redis, microfono o backend usa la modalità debug:

```bash
cd /home/manuel/Scrivania/advX
make frontend
```

Apri <http://localhost:8765/?debug=1>, premi **Start Session** e poi usa i pulsanti
**Right**, **Wrong** e **Low**. Questa modalità permette di verificare tutta
l'interfaccia, le animazioni, gli stati di errore e il feedback fonetico
simulato. I tre pulsanti non compaiono nel prodotto normale.

In alternativa, il backend può servire direttamente gli stessi file:

```bash
make api
```

Apri <http://127.0.0.1:8000/ui/>. Le API interattive sono disponibili su
<http://127.0.0.1:8000/docs>.

### Prova con backend e microfono reali

La prima volta inizializza ambiente, database e benchmark:

```bash
make init
make redis
make build-demo
```

In seguito bastano:

```bash
make redis
make dev
```

Apri <http://localhost:8765>, premi **Start Session**, quindi
tocca il pulsante del microfono:

1. il primo tocco autorizza e avvia la registrazione;
2. pronuncia l'intera frase mostrata;
3. il secondo tocco termina l'acquisizione e richiede la valutazione.

Il browser apre due WebSocket con lo stesso session ID: uno invia PCM mono
16 kHz in frame da 40 ms, l'altro riceve gli eventi di avanzamento e il
risultato finale. La frase viene valutata come frase completa, non come singola
parola isolata: il backend emette aggiornamenti progressivi sui word target
mentre l'utente legge, poi invia un verdetto finale stabile. `localhost` è
considerato un contesto sicuro dai browser; in produzione sono obbligatori
HTTPS e WSS.

La pipeline reale è un MVP tecnico, non uno strumento clinico. Il word gate
attuale rispetta i budget misurati di latenza e false acceptance, ma non il
target di accuratezza: risultati `UNDECIDABLE`, `RETRY` o
`REVIEW_REQUIRED` sono quindi intenzionali.

Per arrestare `make frontend` o `make dev`, premi `Ctrl+C` nel terminale.

## Project status

This project is currently an engineering MVP with an integrated static
frontend, browser microphone streaming, versioned decision pipeline and
production scaffolding. It is not clinically validated.

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
- utterance-final wav2vec2/XLS-R phoneme inference adapter;
- internal CTC Viterbi forced alignment with per-phone spans and scores;
- conservative quality → identity → alignment → phone decision cascade;
- sentence-benchmark streaming with live phrase scoring and per-word updates;
- a runnable local worker path for WAV evaluation;
- static gamified frontend with mock and real-backend modes;
- browser microphone capture, resampling and exact 40 ms PCM streaming;
- authenticated review queue, decision audit trail and Prometheus metrics;
- Docker/Compose/Kubernetes deployment artifacts and health probes;
- unit, integration, black-box, and cleaning-performance tests.

Still to be integrated or validated:

- production-grade adaptive VAD and word-boundary segmentation;
- learner-speech calibration and speaker-disjoint phonetic validation;
- annotated learner-speech validation and clinical review;
- production identity provider and jurisdiction-specific retention controls;
- supported-browser microphone QA and production soak testing.

The phonetic path uses
`facebook/wav2vec2-xlsr-53-espeak-cv-ft` behind `PhonemeModel`, followed by an
internal CTC Viterbi aligner. It runs on a complete isolated word or known
sentence utterance. Its thresholds are explicitly marked uncalibrated until
labelled learner speech is available, so missing or weak evidence yields
`UNDECIDABLE`.

## Architecture

```text
Isolated-word evaluation
    WAV -> cleaning -> MFCC trajectory -> DTW benchmarks -> decision
                                      \-> SQLite attempts and provenance

Live observation
    40 ms PCM frames -> FastAPI WebSocket -> rolling audio window
                                          -> voice-activity progress events
                                          -> background phonetic refreshes
                                          -> stable per-word updates
                                          -> final sentence verdict

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
- eSpeak NG for phonetic scoring
- PortAudio only for the optional command-line microphone tools

Python does not need to be installed manually: `uv sync` downloads the required
Python version, creates `.venv`, and installs the dependencies declared in
`pyproject.toml` at the versions resolved in `uv.lock`.

### CachyOS / Arch Linux

```bash
sudo pacman -S uv espeak-ng portaudio docker
sudo systemctl enable --now docker
```

### macOS

Install [Docker Desktop](https://docs.docker.com/desktop/setup/install/mac-install/)
and uv:

```bash
brew install uv espeak-ng
```

The browser demo does not require `sounddevice`. Install the optional
`microphone` extra only for the command-line microphone tools.

## Setup

From the repository root:

```bash
uv sync --extra phonetic
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
make api
```

Available endpoints:

- `GET /health`
- `GET /health/redis`
- `GET /health/sqlite`
- `GET /metrics`
- `GET /ui/`
- `WS /sessions/{session_id}/events`
- `WS /streaming/sessions/{session_id}`
- interactive FastAPI documentation at `/docs`

Capture and publish microphone frames:

```bash
uv sync --extra phonetic --extra microphone
uv run python -m app.audio_producer
```

The producer generates a server-side UUID and prints it to the terminal. Press
Enter to stop capturing. Frames are published to `audio:{session_id}`.

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

Install the optional phonetic backend. On a new machine, populate its cache
once with the dedicated cache command:

```bash
uv sync --extra phonetic
uv run python -m scripts.cache_phoneme_model \
  --cache-dir data/model-cache
```

The repository's `data/model-cache` is already complete when the sentence
evaluator reports `sentence evaluator: ready`; no download command is then
needed. Start the real phrase UI instead:

```bash
make redis
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 make dev
```

Open <http://localhost:8765>, start the session, and pronounce the complete
sentence displayed by the UI. The `streaming_demo bird` command is a separate
low-level word diagnostic and is not part of this browser flow.

For the WebSocket service, cache the model first and start the API with
`PHONETIC_ENABLED=true`. The server deliberately uses local cached files only;
it never downloads a 1.26 GB model as a side effect of handling a request.

### Browser sentence-pronunciation flow

The browser sends a complete known phrase plus one pedagogical focus word:

```json
{
  "type": "session.start",
  "target_words": ["bird"],
  "expected_text": "A happy bird can fly",
  "locale": "en-US"
}
```

With `expected_text`, the phonetic backend force-aligns and scores **every
phone in the complete phrase**. The target remains the focus word reported in
feedback; it does not limit evaluation to that word. Saying the focus word
inside unrelated or incomplete speech is not sufficient for `CORRECT`.

The runtime does three things in parallel:

1. `pronunciation.progress` reports microphone activity and keeps the listener
   state alive.
2. The phonetic backend refreshes progressive sentence windows in the
   background. When a word boundary is stable enough, it emits
   `stream.inference.partial.word_updates` with the word index, status, score
   and phone-level evidence.
3. The frontend consumes those updates in order, growing the mandala on
   `CORRECT`, shrinking it on `INCORRECT`, and marking low-confidence words
   without pretending the sentence is finished.

Rendering interpolates toward the new layer target on every animation frame, so
inference cadence does not produce visual jumps. Recording ends on a second
microphone tap or after about 1.6 seconds of trailing silence. The final
`pronunciation.evaluated` event is delivered on both sockets and deduplicated by
`attempt_id`, so the result is stable even if one transport arrives late.

The phonetic backend is opt-in because its dependencies and model are large.
Install/cache it as shown above, then set `PHONETIC_ENABLED=true` in `.env`
before running `make api` or `make dev`. With Docker, the image must likewise
include the `phonetic` extra and a local model cache; keep
`PHONETIC_ENABLED=false` in lightweight deployments.

The UI derives `ws://` or `wss://` from the page protocol and uses port 8000 on
the current hostname. To select another API authority:

```text
http://localhost:8765/?api=192.168.1.20:8000
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
app.cleaning.clean_pcm(pcm_s16le: bytes) -> bytes
```

The file-based implementation is named
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
