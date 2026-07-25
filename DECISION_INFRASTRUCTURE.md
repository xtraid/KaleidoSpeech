---
title: "Pronunciation Decision Infrastructure"
subtitle: "Technical Architecture and Clinical Safety Boundary"
author: "advX Speech Service"
date: "24 July 2026"
lang: en-US
geometry: margin=22mm
fontsize: 10pt
colorlinks: true
linkcolor: blue
urlcolor: blue
toc: true
toc-depth: 3
---

# Executive summary

This document explains how the project's decision infrastructure is built,
from raw audio acquisition to clinician-reviewed evidence.

The current system is an experimental English pronunciation assistant. Its
intended role is to help speech-language pathologists collect repeatable
exercises, inspect acoustic evidence, and record their own clinical review. It
is **not** a diagnostic system and does not autonomously prescribe therapy.

The architecture deliberately separates three responsibilities:

1. **Signal and acquisition decisions** determine whether audio is usable.
2. **Acoustic model decisions** compare speech with calibrated word
   benchmarks and expose uncertainty.
3. **Clinical decisions** belong to a qualified human reviewer.

For isolated-word experiments, the model can return `CORRECT`, `INCORRECT`,
`UNDECIDABLE`, or `RETRY`. For natural phrases and pediatric workflows, the
system stores only `REVIEW_REQUIRED` or `RETRY`. This distinction prevents a
word-level adult benchmark from being presented as a validated pediatric
clinical judgment.

# Architectural overview

```text
                     OFFLINE CALIBRATION

 Benchmark WAVs
      |
      v
 Safe ingestion -> cleaning -> temporal features -> speaker-separated splits
      |                                         |
      v                                         v
 SQLite metadata                         DTW medoid prototypes
                                                |
                                                v
                              thresholds + contrastive margins
                                                |
                                                v
                                  versioned active benchmark


                       ONLINE EXECUTION

 Microphone / WAV
      |
      v
 40 ms PCM frames -> frame quality/VAD -> rolling audio window
      |                                      |
      |                                      v
      |                              asynchronous DTW worker
      |                                      |
      v                                      v
 streaming quality events       provisional nearest-word evidence
      \                                      /
       \                                    /
        v                                  v
          session/repetition observation record
                         |
                         v
              speech-language pathologist review
                         |
                         v
                 append-only clinical audit
```

The offline and online paths are intentionally different. Calibration creates
stable, versioned decision artifacts. Online execution consumes those
artifacts but never silently recalibrates them using a patient recording.

# Core data boundaries

## Audio contract

The canonical public audio representation is:

- sample rate: 16,000 Hz;
- channels: one;
- encoding: signed 16-bit PCM, little-endian;
- frame duration: 40 ms;
- samples per frame: 640;
- bytes per frame: 1,280.

Every streaming frame is validated against this exact size. A malformed frame
is rejected rather than padded or truncated because implicit repair would
alter timing and make later evidence difficult to audit.

## Feature contracts

The system currently supports two acoustic representations.

| Representation | Shape | Purpose |
|---|---:|---|
| Summary MFCC embedding | 26 values | Fast baseline and vector retrieval |
| Temporal MFCC sequence | frames × 36 | DTW pronunciation comparison |

The temporal representation uses 12 MFCC coefficients, excluding coefficient
zero, plus first- and second-order temporal derivatives. Utterance-level
normalization reduces broad recording-level scale differences. The feature
and distance implementations carry explicit version identifiers so that
results from incompatible pipelines are not silently mixed.

# Offline benchmark construction

## Safe dataset ingestion

Archive members are validated before extraction to prevent path traversal.
Each source file receives a content hash, and speaker identifiers are
pseudonymized. Duplicate content is ignored idempotently.

The persistent recording row contains the word, locale, pseudonymous speaker,
take number, dataset split, duration, cleaning status, feature version, and
source hash. The original speaker identity is not needed for scoring.

## Speaker-separated splits

Training, validation, and test splits are assigned by speaker rather than by
recording. If recordings from one speaker appeared in both training and test,
the reported accuracy could measure speaker recognition instead of
pronunciation generalization.

The three splits have separate roles:

- **training** selects representative prototypes;
- **validation** calibrates thresholds and margins;
- **test** estimates performance after calibration.

The test split must never be used to choose thresholds.

## Cleaning gate

The cleaning pipeline performs:

- anti-aliased resampling where necessary;
- a 70 Hz high-pass filter;
- energy-based voice activity detection;
- morphological removal of short impulses and filling of short gaps;
- conditional spectral subtraction at low estimated SNR;
- level normalization near -23 dBFS.

Cleaning produces both canonical PCM and a quality report. If the signal is
too short, silent, severely clipped, or otherwise unreliable, the decision
path returns `RETRY`. A quality failure is not interpreted as a pronunciation
error.

## DTW prototypes

For each word, the temporal pipeline selects several real training utterances
as prototypes. It does not average MFCC sequences into an artificial voice.

Prototype selection uses deterministic medoid candidates and a refinement
step:

1. calculate pairwise DTW distances among candidates;
2. select a central candidate;
3. add distant candidates to cover variation;
4. assign candidates to their closest selected prototype;
5. replace each prototype with the most central real recording in its group.

Multiple prototypes are important because pronunciation, pitch, speaking rate,
and vocal-tract characteristics vary substantially across speakers.

## Distance function

For an input temporal sequence $X$ and a prototype $P$, dynamic time
warping finds a low-cost monotonic alignment through their frame-distance
matrix. Frame cost is cosine distance. A band limits excessive warping and
reduces computation.

The distance from an input to a word is:

$$
d(X, w) = \min_{p \in P_w} DTW(X, p)
$$

where $P_w$ is the set of real prototypes for word $w$. Lower distance
means greater acoustic similarity; it does not directly mean clinical
correctness.

# Calibration of the decision regions

Each active word benchmark stores three calibrated values:

- `accept_threshold`;
- `reject_threshold`;
- `margin_threshold`.

Positive validation distances are obtained from recordings of the target
word. Negative distances are obtained from other words. The current prototype
uses conservative quantiles:

$$
T_{accept} = \min(P90(d_{positive}), P10(d_{negative}))
$$

$$
T_{reject} = \max(T_{accept}+0.02, P97(d_{positive}))
$$

The contrastive margin compares the target with the closest competing word:

$$
margin = d(X, competitor) - d(X, target)
$$

A positive margin means that the target is closer. The margin threshold is
calibrated from the lower tail of positive validation margins, with a minimum
safety floor.

These formulas are an experimental calibration policy, not permanent product
constants. A pediatric dataset and an explicit clinical cost model are needed
before selecting production thresholds.

# Isolated-word decision state machine

For an isolated target word, the system computes its distance and the nearest
competitor.

| Condition | State | Meaning |
|---|---|---|
| Audio quality gate fails | `RETRY` | No pronunciation judgment |
| Missing or invalid evidence | `UNDECIDABLE` | The model cannot decide |
| Target distance ≤ accept threshold and margin ≥ margin threshold | `CORRECT` | Accepted by experimental benchmark |
| Target distance ≥ reject threshold | `INCORRECT` | Outside target benchmark |
| Competitor is closer by the required margin | `INCORRECT` | Contrastive mismatch |
| All other valid cases | `UNDECIDABLE` | Evidence falls in the uncertainty region |

The uncertainty region is intentional. Collapsing it into a binary result
would increase either false acceptance or false rejection, and would hide
known limitations of the dataset.

Every result includes reason codes, target distance, competitor identity,
competitor distance, margin, model version, and benchmark version. A user
interface should show a simple message to a child while retaining the detailed
evidence for the clinician and audit log.

# Streaming decision infrastructure

## Why acquisition and model inference are decoupled

The microphone produces one frame every 40 ms. The full DTW comparison can
take longer than this budget, especially when many prototypes are active.
Running it synchronously would block microphone reads and cause frame loss.

The streaming engine therefore has two inference rates:

1. **Every 40 ms:** RMS, peak, clipping, and lightweight voice activity.
2. **Every 200 ms when sufficient voiced audio exists:** schedule a temporal
   comparison on the most recent rolling window.

The DTW executor has one worker and at most one active job. If the worker is
busy, the engine does not queue obsolete windows. Each frame event reports the
most recently completed acoustic evidence and whether a refresh was scheduled
or completed.

On the local 51-frame control recording, `push_frame` took 0.20 ms on average
and 0.75 ms at maximum after asynchronous decoupling. This is below the 40 ms
acquisition budget, but it must be remeasured on deployment hardware and with
the final model.

## Streaming event protocol

The WebSocket endpoint is:

```text
/streaming/sessions/{session_id}
```

The client begins with:

```json
{
  "type": "session.start",
  "target_words": ["bird", "happy"],
  "seed": "demo"
}
```

The server returns a natural or controlled English prompt and the audio
contract. The client then sends binary 1,280-byte frames. Each accepted frame
produces a `stream.inference.partial` event.

The client finishes with:

```json
{"type": "session.finish"}
```

The final event includes duration, voiced-frame ratio, last rolling distances,
model errors if any, and a safe status.

## Phrase-level safety rule

The available benchmarks contain isolated words, while a natural phrase has
coarticulation and no guaranteed pauses between words. A rolling nearest-word
result cannot identify reliable word or phoneme boundaries.

Consequently:

- partial nearest-word evidence is marked `provisional`;
- `clinical_use` is always false;
- `alignment_status` is `not_available`;
- a usable phrase ends in `REVIEW_REQUIRED`;
- insufficient or mostly silent audio ends in `RETRY`.

Reliable phrase feedback requires a forced aligner validated on child speech,
phone-level targets, and evaluation on the expected disorders and age bands.

# Clinical workflow and human authority

The clinical persistence layer models:

- a pseudonymous child profile and age band;
- guardian consent state;
- a clinician-created exercise;
- sound, syllable, word, or phrase level;
- target text and optional target phonemes/error patterns;
- a session and its model version;
- multiple repetitions;
- automatic observations;
- clinician reviews;
- append-only audit events.

The automatic system is technically prevented from writing `CORRECT` or
`INCORRECT` into the pediatric repetition table. It may write only:

- `REVIEW_REQUIRED`;
- `RETRY`.

The clinician separately records:

- `accepted`;
- `speech_error`;
- `retry`;
- structured observations and optional notes.

This separation preserves who made each judgment and supports later
recalibration without rewriting clinical history.

# Persistence architecture

## SQLite: authoritative state

SQLite stores durable, relational information:

- dataset metadata and split assignments;
- processed recording identities;
- temporal features and versioned benchmark artifacts;
- calibrated thresholds;
- attempts and reason codes;
- child consent and exercises;
- sessions, repetitions, reviews, and audit events.

SQLite is appropriate for the current local single-service deployment. WAL
mode, foreign keys, short-lived connections, busy timeouts, unique
constraints, and idempotent ingestion protect basic consistency.

## Redis 8: transport, cache, and optional vector retrieval

Redis 8 is used for transient workloads:

- session event streams;
- benchmark cache;
- transport between producers and consumers;
- optional HNSW nearest-neighbor retrieval.

The v3 runtime is pinned to the official `redis:8.8.0-alpine` image. Redis 8
unifies the former Stack capabilities with the core distribution, including
Redis Search and vector indexing. It also exposes native Vector Set commands.
The deployment chooses Redis's AGPLv3 open-source license option; distribution
and network-service obligations must be reviewed for the final product.

At startup or during diagnostics, `scripts.check_redis` performs read-only
capability discovery. It distinguishes an unreachable service, a reachable
core-only server, and a complete Redis 8 runtime exposing `FT.SEARCH` and
`VSIM`. Failure of optional vector capabilities does not corrupt SQLite state.

`scripts.sync_vector_index` reads accepted 26-dimensional MFCC summaries from
SQLite, computes a locale-wide mean and standard deviation, normalizes every
vector consistently, and upserts it by immutable recording ID. Repeating the
sync updates the same keys. The normalization parameters and version are
stored with the index so query vectors use the identical transform.

Summary embeddings are indexed as 26-dimensional `FLOAT32` vectors with
cosine distance. The vector index accelerates candidate retrieval; it does not
replace SQLite, threshold calibration, temporal DTW, or clinical review.

```text
SQLite                         Redis 8
------                         ------
source of truth                transient acceleration
threshold versions             event streams
clinical audit                 cache
consent and review             vector candidate search
transactional relationships    TTL/session state
```

# Bias and pediatric limitations

The present benchmark fragment is not clinically representative. Known risks
include:

- adult rather than pediatric voices;
- demographic and gender imbalance;
- limited vocabulary;
- no phoneme-level error labels;
- no coverage of speech sound disorders;
- microphone and environment mismatch;
- possible confounding by pitch, accent, or speaking rate.

Observed local failures already demonstrate why uncertainty and human review
are required: a correctly pronounced word by a female speaker can be rejected,
and an acoustically similar non-target pronunciation can be accepted or remain
undecidable.

A vector database does not remove these biases. The correction requires
representative data, speaker-level splits, subgroup metrics, suitable acoustic
representations, calibrated uncertainty, and clinical validation.

# Privacy, security, and retention

Child voice recordings are sensitive personal data. The current clinical row
stores an SHA-256 digest, duration, and observations rather than raw PCM. If
audio must be retained, it should live in a separate encrypted object store
with:

- explicit purpose and legal basis;
- guardian consent and withdrawal workflow;
- role-based access control;
- encryption in transit and at rest;
- retention expiration and verified deletion;
- access and export audit;
- backup deletion policy;
- data protection impact assessment.

Production WebSockets also require authentication, authorization to the
specific session, TLS, origin controls, rate limits, payload limits, and
protection against replay and session identifier guessing.

# Failure modes and safe behavior

| Failure | Safe behavior |
|---|---|
| Wrong frame size | Reject frame and report protocol error |
| Silence or very short recording | `RETRY` |
| Cleaning cannot produce reliable speech | `RETRY` |
| Target benchmark missing | Explicit error; no fallback judgment |
| Model job exceeds frame interval | Continue acquisition; keep last evidence |
| Model worker throws an exception | Report `model_error`; do not invent result |
| Distances fall between thresholds | `UNDECIDABLE` |
| Phrase has no validated alignment | `REVIEW_REQUIRED` |
| Consent is absent or withdrawn | Do not create exercise/session |
| Clinician has not reviewed | Preserve automatic observation only |
| Redis unavailable | Durable SQLite data remains authoritative |

# Verification strategy

The repository currently has 74 passing tests. Relevant decision tests cover:

- cleaning contract and quality behavior;
- SQLite idempotency and transactions;
- Redis/Valkey protocol doubles and fuzzing;
- acoustic repository constraints;
- temporal feature and DTW behavior;
- contrastive decision logic;
- exact 40 ms streaming frames;
- non-blocking streaming refresh behavior;
- safe phrase final state;
- consent-gated clinical persistence;
- separation between system and clinician verdicts;
- append-only audit records;
- vector index command and shape validation.

Before clinical use, software tests must be complemented by a locked external
evaluation dataset and prospective human validation.

# Recommended evolution

## Phase 1 — engineering prototype

- Keep word-level DTW and explicit uncertainty.
- Add frame-loss, jitter, and long-session stress tests.
- Add authentication and encrypted audio storage only if retention is needed.
- Instrument end-to-end latency and model queue saturation.

## Phase 2 — speech analysis

- Define a phoneme inventory and clinician-approved error taxonomy.
- Integrate an open forced aligner suitable for English.
- Store aligned evidence with model and alignment versions.
- Evaluate phone-level observations without presenting diagnoses.

## Phase 3 — pediatric calibration

- Acquire legally usable pediatric and disordered-speech corpora.
- Split strictly by child.
- Report sensitivity, specificity, false rejection, false acceptance, and
  calibration by age band, gender, accent, microphone, and error type.
- Define operating thresholds with clinicians and document the cost of each
  error.

## Phase 4 — supervised product trial

- Complete privacy and regulatory assessment.
- Freeze model and benchmark versions for the trial.
- Run usability and safety studies with speech-language pathologists.
- Measure whether the tool improves workflow without replacing professional
  judgment.

# Source map

The principal implementation files are:

| Responsibility | File |
|---|---|
| Audio cleaning adapter | `app/cleaning.py` |
| Signal processing | `app/audio_cleaning.py` |
| Temporal features and DTW | `app/temporal_features.py` |
| Prototype calibration and decisions | `app/temporal_benchmark.py` |
| Temporal persistence | `app/temporal_repository.py` |
| 40 ms streaming engine | `app/streaming_inference.py` |
| WebSocket protocol | `app/api.py` |
| Prompt generation | `app/sentence_composer.py` |
| Pediatric clinical persistence | `app/clinical_repository.py` |
| Redis 8 vector index | `app/vector_index.py` |
| SQLite-to-HNSW synchronization | `app/vector_sync.py` |
| SQLite schema | `scripts/init_db.py` |
| Microphone demonstration | `scripts/streaming_demo.py` |

# Conclusion

The decision infrastructure is designed around traceability and uncertainty.
It combines deterministic signal checks, versioned acoustic evidence,
contrastive DTW decisions, non-blocking streaming, relational audit, and an
explicit human clinical boundary.

The most important next improvement is not a more aggressive threshold or a
different database. It is representative pediatric, phone-labelled data and a
validated alignment/evaluation protocol designed with speech-language
pathologists. Until then, the architecture can support controlled experiments
and clinician review, but should not make autonomous clinical claims.
