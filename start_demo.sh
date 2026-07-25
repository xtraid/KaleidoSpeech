#!/usr/bin/env bash

set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
FIRST_RUN_MARKER="$ROOT_DIR/.venv/.demo-initialized"

cd "$ROOT_DIR"

log() {
  printf '\n\033[1;36m%s\033[0m\n' "$1"
}

fail() {
  printf '\nErrore: %s\n' "$1" >&2
  exit 1
}

# Installazione e inizializzazione: questa sezione viene eseguita una sola volta.
if [[ ! -f "$FIRST_RUN_MARKER" ]]; then
  log "Prima inizializzazione della demo"

  command -v brew >/dev/null 2>&1 || fail \
    "Homebrew non è installato. Installalo da https://brew.sh e riesegui lo script."

  if ! command -v uv >/dev/null 2>&1; then
    log "Installazione di uv"
    HOMEBREW_NO_AUTO_UPDATE=1 brew install uv
  fi

  if ! HOMEBREW_NO_AUTO_UPDATE=1 brew list --versions espeak-ng >/dev/null 2>&1; then
    log "Installazione di eSpeak NG"
    HOMEBREW_NO_AUTO_UPDATE=1 brew install espeak-ng
  fi

  if ! command -v docker >/dev/null 2>&1; then
    log "Installazione di Docker Desktop"
    HOMEBREW_NO_AUTO_UPDATE=1 brew install --cask docker
  fi

  if [[ ! -f .env ]]; then
    cp .env.example .env
  fi

  # Abilita il modello fonetico senza duplicare la variabile nel file.
  if grep -q '^PHONETIC_ENABLED=' .env; then
    sed -i.bak 's/^PHONETIC_ENABLED=.*/PHONETIC_ENABLED=true/' .env
    rm -f .env.bak
  else
    printf '\nPHONETIC_ENABLED=true\n' >>.env
  fi

  log "Installazione delle dipendenze Python e inizializzazione del database"
  make init

  log "Download del modello fonetico"
  uv run python -m scripts.cache_phoneme_model --cache-dir data/model-cache

  # Il marker viene creato soltanto se tutti i passaggi precedenti hanno successo.
  touch "$FIRST_RUN_MARKER"
  log "Prima inizializzazione completata"
fi

command -v docker >/dev/null 2>&1 || fail "Docker non è disponibile."

if ! docker info >/dev/null 2>&1; then
  log "Avvio di Docker Desktop"
  open -gja Docker || fail "Impossibile avviare Docker Desktop."

  for _ in {1..60}; do
    docker info >/dev/null 2>&1 && break
    sleep 2
  done

  docker info >/dev/null 2>&1 || fail \
    "Docker Desktop non è pronto. Aprilo manualmente e riesegui lo script."
fi

log "Avvio di Redis"
make redis

ESPEAK_PREFIX="$(
  HOMEBREW_NO_AUTO_UPDATE=1 brew --prefix espeak-ng
)"
ESPEAK_LIBRARY="$ESPEAK_PREFIX/lib/libespeak-ng.dylib"

[[ -f "$ESPEAK_LIBRARY" ]] || fail \
  "Libreria eSpeak non trovata in $ESPEAK_LIBRARY."

export PHONEMIZER_ESPEAK_LIBRARY="$ESPEAK_LIBRARY"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

log "Demo disponibile su http://localhost:8765 (Ctrl+C per arrestarla)"
exec make dev
