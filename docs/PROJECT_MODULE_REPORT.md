# advX — report dei moduli e guida di test

## 1. Scopo attuale

advX è un prototipo per valutare la pronuncia di audio mono a 16 kHz. Il
percorso MVP attuale riceve una frase nota, trasporta il microfono in frame da
40 ms attraverso Redis, allinea progressivamente l'audio a un benchmark
temporale annotato e produce:

- feedback di ascolto;
- posizione corrente nella frase;
- un evento quando una porzione locale supera la soglia di errore;
- un giudizio finale sull'intera frase;
- un tentativo persistito in SQLite.

Il repository contiene anche il precedente percorso sperimentale per parole
isolate. I due percorsi condividono audio, Redis, SQLite e feature temporali,
ma non devono essere confusi.

### Percorso MVP frase

```text
scripts.sentence_streaming_demo
    → app.redis_bus.publish_audio_frame
    → Redis Stream audio:{session_id}
    → app.worker.parse_audio_frame
    → app.sentence_streaming.SentenceStreamingSession
    → app.sentence_repository.save_sentence_attempt
    → app.redis_bus.publish_ui_event
    → Redis Stream session:{session_id}:events
```

### Percorso sperimentale parola isolata

```text
ZIP/WAV
    → dataset_ingest / cleaning
    → acoustic_features e temporal_features
    → acoustic_benchmark / temporal_benchmark
    → repository SQLite
    → worker oppure streaming_demo
```

## 2. Contratti condivisi

### Audio

- PCM signed 16-bit little-endian;
- mono;
- 16.000 Hz;
- 40 ms per frame;
- 640 campioni e 1.280 byte per frame.

Ogni record Redis audio contiene:

```text
sequence
captured_at_ns
sample_rate
frame_ms
pcm_s16le
```

### Esiti

- `CORRECT`: evidenza entro la soglia del benchmark;
- `INCORRECT`: evidenza incompatibile con il benchmark;
- `UNDECIDABLE`: evidenza valida ma nella fascia incerta, usato dal percorso
  parola;
- `RETRY`: audio insufficiente o non valutabile.

### Persistenza

SQLite è lo stato durevole. Redis contiene trasporto audio, eventi, cache e
stato effimero della sessione.

## 3. Moduli `app`

### `app/__init__.py`

Marca `app` come package Python. Non contiene logica applicativa.

### `app/config.py`

Definisce `Settings` e legge la configurazione dalle variabili d'ambiente.
Calcola inoltre campioni e byte per frame.

Variabili principali:

| Variabile | Default | Significato |
|---|---:|---|
| `APP_HOST` | `127.0.0.1` | indirizzo FastAPI |
| `APP_PORT` | `8000` | porta FastAPI |
| `SQLITE_PATH` | `data/pronunciation.sqlite3` | database SQLite |
| `REDIS_URL` | `redis://localhost:6379/0` | connessione Redis |
| `SAMPLE_RATE` | `16000` | frequenza audio |
| `FRAME_MS` | `40` | durata del frame |
| `SESSION_TTL_SECONDS` | `900` | TTL dello stato Redis |
| `MAX_STREAM_LENGTH` | `2000` | limite indicativo dello stream audio |

`get_settings()` è memorizzata in cache. Nei test che cambiano variabili
d'ambiente bisogna chiamare `get_settings.cache_clear()`.

### `app/database.py`

Espone `db_connection()`, un context manager per connessioni SQLite brevi.
Attiva foreign key e busy timeout e restituisce righe indicizzabili per nome.

### `app/redis_bus.py`

È il confine di accesso a Redis:

- crea il client condiviso;
- verifica raggiungibilità, Redis Search e Vector Set;
- legge e scrive cache JSON con TTL;
- pubblica frame su `audio:{session_id}`;
- aggiorna `session:{session_id}:state`;
- pubblica eventi su `session:{session_id}:events`.

Il percorso frase usa Redis Streams per ogni frame e per ogni evento.

### `app/audio_producer.py`

Acquisisce il microfono attraverso `sounddevice`. La callback mette i frame in
una coda locale; un thread separato li pubblica su Redis per non eseguire I/O di
rete nella callback audio.

`split_pcm()` è anche usata nei test per dividere PCM già disponibile.

Questo producer generico non avvia da solo un valutatore. La demo frase integra
acquisizione e valutazione.

### `app/audio_cleaning.py`

Implementa la pipeline completa di pulizia per file WAV:

- scelta/conversione mono;
- ricampionamento a 16 kHz;
- filtro passa-alto;
- rilevamento delle regioni vocali;
- denoise spettrale;
- normalizzazione e report di qualità.

`CleaningReport` descrive stato, livelli e avvisi. È principalmente usato
dall'importazione offline e dalla valutazione di parole isolate.

### `app/cleaning.py`

Adatta la pipeline a PCM in memoria. Scrive temporaneamente WAV, invoca
`clean_wav_files()` e restituisce nuovamente PCM canonico.

- `clean_pcm_with_report()` restituisce audio e report;
- `clean_pcm()` restituisce solo l'audio;
- `clean_audio` è un alias compatibile con il vecchio contratto.

### `app/temporal_features.py`

Estrae una sequenza temporale da PCM:

- frame acustici da 25 ms;
- hop da 10 ms;
- mel filter bank;
- 12 coefficienti cepstrali;
- delta e delta-delta;
- dimensione finale 36.

Contiene anche `dtw_distance()`, DTW con costo coseno, normalizzazione per
lunghezza e banda di Sakoe-Chiba.

È la base acustica condivisa dal percorso DTW per parole e dal benchmark di
frase.

### `app/sentence_streaming.py`

È il nucleo dell'MVP frase.

`SentenceUnit`
: parola o unità annotata con `start_ms` ed `end_ms`.

`SentenceBenchmark`
: testo, lingua, locale, feature del riferimento, timeline e soglie.

`extract_streaming_features()`
: divide sia benchmark sia audio live in chunk identici da 200 ms. Questa
simmetria evita differenze di preprocessing tra costruzione e inferenza.

`OnlineSentenceAligner`
: conserva una sola riga della matrice DTW e la aggiorna con le nuove feature.
La posizione migliore viene convertita nell'unità temporale corrente.

`SentenceStreamingSession`
: accetta esattamente un frame da 40 ms alla volta. Ogni 200 ms aggiorna
l'allineamento. Il VAD del singolo frame alimenta il feedback immediato; il VAD
del chunk completo decide se conservare l'inizio della frase.

Un errore locale deve superare `error_threshold` per due chunk consecutivi.
Con i valori correnti ciò introduce circa 400 ms di latenza di conferma.
L'evento contiene:

```json
{
  "type": "pronunciation.error",
  "unit": "bird",
  "unit_index": 2,
  "start_ms": 600,
  "detected_at_ms": 1000,
  "status": "INCORRECT"
}
```

`start_ms` è una stima retrospettiva; `detected_at_ms` è il momento in cui
l'evidenza diventa sufficiente.

### `app/sentence_repository.py`

Persiste e ricostruisce i benchmark di frase:

- serializza le feature NumPy senza pickle;
- salva timeline e metadati;
- disattiva versioni precedenti della stessa frase;
- carica il benchmark attivo per `sentence_id`;
- salva risultato finale ed eventi di errore in `sentence_attempts`.

### `app/dataset_ingest.py`

Importa in sicurezza registrazioni di parole contenute in ZIP:

- rifiuta path traversal e membri anomali;
- limita dimensione e rapporto di compressione;
- decodifica WAV senza estrarlo;
- pseudonimizza il parlante;
- assegna split deterministici per parlante;
- calcola indicatori di qualità.

È attualmente specifico per il formato del dataset di parole isolate.

### `app/acoustic_features.py`

Vecchia baseline per parole isolate. Riduce una registrazione a un vettore
MFCC riassuntivo di 26 valori. Rimane per confronto e per l'indice vettoriale,
ma non è usato dal nuovo MVP frase.

### `app/acoustic_benchmark.py`

Gestisce il percorso offline/online della baseline summary:

- importa ZIP;
- deduplica prima di applicare `limit_per_word`;
- pulisce e salva feature;
- costruisce prototipi e soglie;
- valuta una parola isolata.

Durante l'importazione salva anche feature temporali, usate dal benchmark DTW.

### `app/acoustic_repository.py`

Repository SQLite per registrazioni elaborate, benchmark summary e tentativi
summary. Contiene conversione tra record SQL, dataclass e matrici NumPy.

### `app/temporal_benchmark.py`

Costruisce e valuta benchmark DTW di parole isolate:

- seleziona medoid reali;
- calcola distanze intra-classe e verso concorrenti;
- calibra soglie e margine;
- produce decisioni contrastive;
- salva il tentativo.

Non effettua l'allineamento della frase: quello appartiene a
`sentence_streaming.py`.

### `app/temporal_repository.py`

Repository delle feature temporali e dei benchmark DTW di parola. Comprime le
sequenze NumPy, salva prototipi, soglie, versioni e tentativi.

### `app/decision.py`

Policy deterministica della baseline summary. Trasforma distanza e qualità in
`CORRECT`, `INCORRECT`, `UNDECIDABLE` o `RETRY`, mantenendo reason code
espliciti.

### `app/benchmark_repository.py`

Repository del contratto storico basato su fonemi attesi. Carica benchmark
lessicali e salva tentativi generici. È usato dal vecchio worker e dai relativi
test.

### `app/pronunciation_engine.py`

Definisce soltanto il protocollo di un futuro motore fonetico e la struttura
`PronunciationResult`. Non contiene un modello concreto.

### `app/streaming_inference.py`

Prototipo streaming precedente per parole:

- genera un prompt;
- calcola VAD ogni 40 ms;
- mantiene una finestra mobile;
- avvia DTW periodico in background;
- restituisce osservazioni provvisorie.

Non localizza errori nella frase ed è distinto da `sentence_streaming.py`.

### `app/sentence_composer.py`

Genera deterministicamente frasi inglesi contenenti le parole richieste. Serve
al prototipo precedente; il nuovo MVP riceve una frase benchmark già definita
e non usa questo modulo.

### `app/vector_index.py`

Indice HNSW opzionale su Redis Search per i vettori summary a 26 dimensioni:

- crea l'indice;
- inserisce embedding e metadati;
- esegue KNN filtrato obbligatoriamente per locale.

Non è necessario per confrontare una frase con un `sentence_id` noto.

### `app/vector_sync.py`

Sincronizza in Redis le feature summary autorevoli presenti in SQLite.
Calcola normalizzazione separata per ogni locale e salva chiavi distinte, per
evitare confronti tra spazi normalizzati diversamente.

### `app/clinical_repository.py`

Persistenza opzionale per client pediatrici, esercizi, sessioni, ripetizioni,
review e audit. Non partecipa al calcolo acustico e non viene usato dalla demo
MVP.

### `app/worker.py`

Contiene primitive del worker Redis storico:

- valida i record audio;
- verifica contiguità della sequenza;
- gestisce consumer group e pending message;
- carica benchmark con cache Redis;
- persiste e pubblica risultati.

Espone anche la CLI `evaluate-wav` per parole isolate. La demo frase riusa
`parse_audio_frame()` per assicurarsi che il frame valutato sia quello
effettivamente trasportato da Redis.

### `app/api.py`

Espone:

- `GET /health`;
- `GET /health/redis`;
- WebSocket degli eventi di sessione;
- WebSocket del vecchio prototipo streaming.

Il WebSocket streaming non è ancora collegato a
`SentenceStreamingSession`. Per testare l'MVP frase bisogna usare
`scripts.sentence_streaming_demo`.

## 4. Moduli `scripts`

### `scripts/init_db.py`

Crea in modo idempotente tabelle e indici SQLite. Conserva eventuali vecchie
tabelle incompatibili rinominandole `legacy_*`.

Per il percorso frase crea:

- `sentence_benchmarks`;
- `sentence_attempts`.

### `scripts/build_sentence_benchmark.py`

Costruisce un benchmark di frase da:

- `sentence_id`;
- testo esatto;
- WAV mono 16 kHz;
- JSON con timeline delle unità;
- lingua, locale e soglie opzionali.

Il WAV viene troncato all'ultimo chunk completo di 200 ms.

### `scripts/sentence_streaming_demo.py`

È la demo end-to-end del nuovo MVP:

1. apre il microfono;
2. pubblica ciascun frame su Redis;
3. rilegge il frame dallo stream;
4. lo valida;
5. aggiorna la sessione;
6. pubblica gli eventi su Redis;
7. stampa gli errori;
8. salva il risultato finale.

### `scripts/build_acoustic_benchmarks.py`

Importa gli ZIP di parole isolate e costruisce sia benchmark summary sia
benchmark temporali.

### `scripts/demo_evaluate.py`

Valuta un WAV di parola isolata usando il benchmark DTW già costruito.

### `scripts/streaming_demo.py`

Demo microfono del precedente percorso a parole. Con `--decision-only` valuta
una singola parola dopo il silenzio finale.

### `scripts/check_redis.py`

Verifica raggiungibilità e capacità Redis. Richiede Redis Search perché è stata
scritta anche per il percorso vettoriale opzionale.

### `scripts/sync_vector_index.py`

Popola l'indice HNSW opzionale e può interrogare i vicini di una registrazione.
Normalizzazione e ricerca sono isolate per locale.

### `scripts/__init__.py`

Marca `scripts` come package eseguibile con `python -m`.

## 5. Database

Le tabelle appartengono a quattro gruppi.

| Gruppo | Tabelle principali |
|---|---|
| Frasi MVP | `sentence_benchmarks`, `sentence_attempts` |
| Parole temporali | `processed_recordings`, `temporal_features`, `temporal_benchmarks`, `temporal_attempts` |
| Baseline summary | `acoustic_benchmarks`, `acoustic_attempts` |
| Storico/clinico | `benchmarks`, `attempts`, `pediatric_clients`, `therapy_exercises`, `clinical_sessions`, `clinical_repetitions`, `clinician_reviews`, `clinical_audit_events` |

Per ispezionare le frasi:

```bash
sqlite3 data/pronunciation.sqlite3 "SELECT sentence_id,text,locale,version,active FROM sentence_benchmarks;"
```

Per ispezionare i tentativi:

```bash
sqlite3 data/pronunciation.sqlite3 "SELECT session_id,status,distance,errors_json,created_at FROM sentence_attempts ORDER BY id DESC LIMIT 10;"
```

## 6. Guida rapida di installazione

Dal root del progetto:

```bash
cd /home/manuel/Scrivania/advX
```

Installa le dipendenze:

```bash
uv sync
```

Inizializza SQLite:

```bash
uv run python -m scripts.init_db
```

Avvia Redis:

```bash
docker compose up -d redis
```

Controlla Redis:

```bash
docker compose ps
```

Per il solo percorso frase sono sufficienti Streams e persistenza Redis. Il
comando `scripts.check_redis` controlla anche Redis Search, necessario soltanto
per la funzionalità vettoriale opzionale:

```bash
uv run python -m scripts.check_redis
```

Controlla il microfono con un comando su una sola riga:

```bash
uv run python -c "import sounddevice as sd; print(sd.query_devices())"
```

## 7. Preparazione di un benchmark di frase

### 7.1 WAV

Il file deve essere:

- mono;
- 16 kHz;
- PCM leggibile da `soundfile`;
- pronunciato correttamente;
- senza lunghi silenzi iniziali o finali.

Controllo:

```bash
uv run python -c "import soundfile as sf; x,sr=sf.read('benchmark.wav'); print({'sample_rate':sr,'shape':x.shape,'duration_s':len(x)/sr})"
```

Se necessario, con FFmpeg:

```bash
ffmpeg -i input.wav -ac 1 -ar 16000 -c:a pcm_s16le benchmark.wav
```

### 7.2 Timeline

Crea `units.json`:

```json
[
  {"text": "the", "start_ms": 0, "end_ms": 180},
  {"text": "happy", "start_ms": 180, "end_ms": 520},
  {"text": "bird", "start_ms": 520, "end_ms": 850}
]
```

Regole:

- tempi in millisecondi rispetto all'inizio del WAV;
- `end_ms` strettamente maggiore di `start_ms`;
- unità nell'ordine della frase;
- l'ultima unità deve coprire la parte finale pronunciata;
- per ora la timeline viene fornita manualmente o da forced alignment offline.

### 7.3 Costruzione

```bash
uv run python -m scripts.build_sentence_benchmark en-001 "The happy bird" benchmark.wav units.json
```

Con soglie esplicite:

```bash
uv run python -m scripts.build_sentence_benchmark en-001 "The happy bird" benchmark.wav units.json --locale en-US --error-threshold 0.55 --final-threshold 0.40 --version 1
```

Output atteso:

```json
{"benchmark_id": 1, "sentence_id": "en-001", "units": 3}
```

Ricostruire la stessa versione aggiorna il benchmark. Una nuova versione
disattiva quella precedente.

## 8. Test end-to-end streaming

Assicurati che Redis sia attivo, poi:

```bash
uv run python -m scripts.sentence_streaming_demo en-001 --duration 8
```

La demo mostra la frase, attende Invio e registra per la durata indicata.

Output indicativo:

```text
00200 ms  the  ASCOLTO
00400 ms  happy  ASCOLTO
01000 ms  ERRORE su «bird» (iniziato a 600 ms)
```

Risultato finale:

```json
{
  "type": "sentence.evaluated",
  "sentence_id": "en-001",
  "status": "INCORRECT",
  "distance": 0.48,
  "threshold": 0.4,
  "duration_ms": 8000,
  "model_version": "known-phrase-online-dtw-v1",
  "attempt_id": 1
}
```

Per adattare il rumore ambientale:

```bash
uv run python -m scripts.sentence_streaming_demo en-001 --duration 8 --voice-threshold 0.03
```

Indicazioni pratiche:

- se non parte l'allineamento, abbassa gradualmente la soglia;
- se il rumore viene interpretato come voce, alzala;
- pronuncia la frase subito dopo Invio;
- usa una durata poco superiore alla frase;
- benchmark e tentativo devono usare lo stesso microfono/ambiente soltanto per
  il primo smoke test, non per la validazione reale.

## 9. Controllo degli eventi Redis

Durante o dopo la demo:

```bash
redis-cli --scan --pattern 'audio:*'
```

```bash
redis-cli --scan --pattern 'session:*:events'
```

Dato un `session_id` stampato dalla demo:

```bash
redis-cli XRANGE "session:SESSION_ID:events" - +
```

Gli eventi importanti sono:

- `sentence.listening`, ogni 40 ms;
- `sentence.alignment`, ogni 200 ms;
- `pronunciation.error`, dopo conferma;
- `sentence.evaluated`, alla fine.

## 10. Test automatici

Suite completa:

```bash
uv run pytest -q
```

Stato verificato alla creazione di questo report:

```text
87 passed
```

Test del nuovo MVP frase:

```bash
uv run pytest tests/test_sentence_streaming.py -q
```

Test Redis e ricerca vettoriale:

```bash
uv run pytest tests/test_redis_capabilities.py tests/test_vector_index.py tests/test_vector_sync.py -q
```

Test importazione e benchmark:

```bash
uv run pytest tests/test_acoustic_decision_pipeline.py tests/test_acoustic_repository.py tests/test_temporal_dtw.py -q
```

Test audio cleaning:

```bash
uv run pytest tests/test_audio_cleaning.py tests/test_cleaning_black_box.py tests/test_cleaning_performance.py -q
```

Test API e database:

```bash
uv run pytest tests/test_api.py tests/test_database_black_box.py tests/test_clinical_repository.py -q
```

Verifica sintattica:

```bash
uv run python -m compileall -q app scripts tests
```

## 11. Test della vecchia modalità parola

Costruisci i benchmark inclusi:

```bash
uv run python -m scripts.build_acoustic_benchmarks --limit-per-word 100 tmp/1.zip tmp/2.zip tmp/3.zip
```

Valuta un WAV mono 16 kHz:

```bash
uv run python -m app.worker evaluate-wav bird /percorso/reale/audio.wav
```

Demo microfono:

```bash
uv run python -m scripts.streaming_demo bird --duration 8 --decision-only --voice-threshold 0.03
```

Questa modalità non localizza errori dentro una frase.

## 12. Test API

Avvio:

```bash
uv run uvicorn app.api:app --host 127.0.0.1 --port 8000 --reload
```

Health:

```bash
curl http://127.0.0.1:8000/health
```

Redis health:

```bash
curl http://127.0.0.1:8000/health/redis
```

Il WebSocket del nuovo percorso frase non è ancora esposto dall'API. La demo
CLI è attualmente il test end-to-end autorevole.

## 13. Problemi comuni

### `Format not recognised`

Il percorso indicato non è un WAV valido. Controllare con `soundfile` o
convertire con FFmpeg. Non usare `/percorso/audio.wav` letteralmente.

### `Redis is unreachable`

```bash
docker compose up -d redis
```

Poi:

```bash
redis-cli ping
```

L'output deve essere `PONG`.

### `No sentence benchmark`

Eseguire prima `scripts.init_db` e `scripts.build_sentence_benchmark`, usando
lo stesso `sentence_id` passato alla demo.

### La sessione resta in ascolto

Misurare il rumore del microfono e regolare `--voice-threshold`. Il VAD del
chunk usa tutti i 200 ms, quindi una pausa negli ultimi 40 ms non elimina più
l'inizio parlato.

### Molti falsi errori

Le soglie predefinite non sono ancora calibrate sul database reale. Verificare
anche timeline, silenzi del benchmark, locale, microfono e velocità. Non
abbassare arbitrariamente la soglia per far passare un singolo parlante:
calibrarla su train/validation e misurarla su speaker non presenti nel training.

## 14. Limiti attuali e prossimo lavoro

- un benchmark di frase contiene attualmente un solo riferimento acustico;
- le soglie di default sono provvisorie;
- la timeline deve essere fornita offline;
- l'errore è localizzato a livello di unità annotata, normalmente parola;
- il modello misura somiglianza acustica, non diagnostica l'errore fonetico;
- la conferma live ha circa 400 ms di ritardo;
- il WebSocket API usa ancora il vecchio motore streaming;
- la demo rilegge Redis nello stesso processo, mentre un deployment reale
  separerà producer, worker e client;
- autenticazione, consenso, retention e protezione dei dati non sono ancora
  pronti per produzione.

Con il database completo, i passi successivi sono:

1. importare più registrazioni corrette per frase;
2. generare o verificare automaticamente le timeline;
3. scegliere prototipi rappresentativi;
4. calibrare soglia locale e finale su validation;
5. misurare falsi positivi e falsi negativi su speaker separati;
6. collegare `SentenceStreamingSession` al WebSocket e a un worker Redis
   indipendente.
