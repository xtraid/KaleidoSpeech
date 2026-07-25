# Phoneme model decision

**Decision:** server-side
`facebook/wav2vec2-xlsr-53-espeak-cv-ft`, pinned to revision
`2c733782da5604684829819a5eb744c193fe9398`.

The checkpoint exposes frame-level phonetic CTC evidence, is multilingual, and
uses an eSpeak-compatible inventory. The adapter is lazy and optional:
`PHONETIC_ENABLED=false` keeps the lightweight service operational, while
`uv sync --extra phonetic` installs the model runtime. Model downloads are not
performed implicitly. Run
`uv run python scripts/cache_phoneme_model.py --cache-dir data/model-cache`
during the image/artifact build; runtime inference remains offline-only and
loads the exact pinned revision from that cache.

The model runs on the server because browser deployment would increase client
memory, expose inconsistent hardware performance, and complicate artifact
versioning.

## Budgets and gates

- startup/model load under 5 seconds from a warm cache;
- inference under 200 ms for one second of 16 kHz audio on target hardware;
- end-to-end response under 500 ms;
- measure resident memory on CPU and GPU deployment targets;
- validate token coverage for every lexicon pronunciation;
- validate accuracy separately on pediatric/learner speech before enabling
  automatic feedback.

Apache-2.0 compatibility and the exact upstream model card/license must be
rechecked as part of release review.
