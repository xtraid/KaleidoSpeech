# Roadmap implementation status

Updated: 2026-07-24

## Implemented

- configurable VAD threshold and validated environment settings;
- optional WebSocket bearer-token authentication;
- bounded control messages and validated session/target input;
- per-client HTTP and per-session WebSocket sliding-window rate limits;
- JSON request logging with correlation IDs;
- Redis and SQLite health probes and WebSocket shutdown handling;
- closed-vocabulary word identity contract and DTW baseline;
- pinned, lazy phoneme-model adapter and versioned lexicon;
- NumPy CTC forced alignment, phoneme scoring, conservative decision cascade;
- stable public modules for word identity, phoneme model, alignment, scoring,
  and the ML-ready decision engine;
- frontend integration and model-decision documentation.
- word-gate and per-phone calibration utilities, joint Bayesian search, and
  immutable threshold configuration;
- Redis circuit breaker with bounded exponential retry;
- protected human-review queue and ground-truth annotation endpoints;
- Italian/English feedback, evaluated events and latency histograms;
- Postman, Kubernetes, TLS/proxy, backup and child-data security artifacts.
- imported static frontend with mock workflow, safe real-backend switch,
  Makefile launch targets, `/ui/` serving and a tested flat event adapter.
- browser microphone capture and exact PCM streaming to the real backend;
- vectorized/downsampled DTW with measured validation p95 63.55 ms and
  wrong-word pair FAR 1.953%;
- versioned ML feature schema, guarded Random Forest trainer, A/B assignment,
  PSI drift utility and model activation/rollback registry.

## Blocked on empirical work

Accuracy, false acceptance/rejection, latency, memory, and pediatric suitability
cannot be marked complete from source code. They require a speaker-disjoint
labeled corpus, target deployment hardware, a domain expert, calibration, and a
frozen test set. Random Forest work additionally requires at least 500 reviewed
attempts. These remain release gates rather than fabricated pass results.

The existing validation corpus is sufficient only for word identity: it
contains 64 persisted acceptable temporal recordings. The calibrated DTW gate
meets isolated FAR and latency budgets but only matches 18.75% of correct
validation words because 73.44% are conservatively `UNCERTAIN`; it therefore
fails the >95% MVP accuracy gate.

## Remaining engineering

- generated frontend client and validation against the actual frontend;
- calibrated threshold values from a real speaker-disjoint corpus;
- distributed rate limiting/circuit state for multi-instance deployment;
- production alert rules and infrastructure-specific secrets/TLS manifests;
- forced-alignment insertion/deletion and duration-error validation against
  annotated learner speech.
