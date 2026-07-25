# Pronunciation UI

Self-contained single-file prototype with no build step, dependencies, or
framework. Open `index.html` in a browser or serve it with any static server.

## Quick start

```bash
cd frontend
python3 -m http.server 8765
# open http://localhost:8765
```

## What's in this folder

```
index.html    Single-file app (HTML + CSS + JS, including scene images)
README.md     This file
```

## How it works

The app is a gamified full-sentence pronunciation screen built around a
kaleidoscope. Each level displays a short sentence and identifies one
pedagogical focus word. The child records the complete sentence; the backend
scores every phone in it, streams stable `stream.inference.partial.word_updates`
while the sentence is still being read, and then sends a final
`pronunciation.evaluated` event. A successful full-phrase result completes the
level.

## Real mode (default)

Open the page normally to connect to the real backend and use the large
microphone button. The first tap begins recording; the second stops and submits
the utterance. While speaking, the UI keeps the listening state alive from
`pronunciation.progress` and applies per-word updates as soon as they become
stable enough to trust.

## Debug/mock mode

Open `http://localhost:8765/?debug=1`. Only in this explicit mode:

- Clicking **Start Session** simulates a successful WebSocket connection after 800 ms.
- Three mock buttons appear at the bottom: **Wrong**, **Low** (low confidence), **Right**.
- Keyboard shortcuts also work: `R`/`ArrowRight` = correct, `W`/`ArrowLeft` = wrong, `L` = low confidence, `C` = Chinese tone error.
- No real WebSocket or microphone is used.

The **Right**, **Wrong** and **Low** controls are intentionally absent from the
normal product UI.

## Backend integration

### WebSocket

The app opens event and audio-stream connections:

```
ws://{current-host}:8000/sessions/{session_id}/events
ws://{current-host}:8000/streaming/sessions/{session_id}
```

The `session_id` is generated client-side (random, no personal data). The app opens one connection per session and closes it when leaving.
Under HTTPS it automatically uses `wss://`. A separate backend can be selected
with `?api=host:port`.

### Events consumed

| Event type | What the UI does |
|---|---|
| `pronunciation.progress` | Updates the listening indicator |
| `stream.inference.partial.word_updates` | Applies stable per-word phonetic results while the sentence is being read |
| `pronunciation.evaluated` | Shows the full-phrase result, updates the kaleidoscope, and renders phone-level feedback |

### Evaluation logic

The UI uses the evaluated event in this order:

1. If `status` is present, it wins. `CORRECT` grows the mandala, `INCORRECT`
   shrinks it, and `RETRY`, `UNDECIDABLE`, or `REVIEW_REQUIRED` switch to the
   low-confidence flow.
2. If `status` is missing, the UI falls back to `confidence`. Values below
   `CONFIG.confidenceThreshold` (default 0.6) are treated as low confidence, not
   as a wrong answer.
3. If confidence is sufficient, `score >= 0.7` is treated as correct and
   anything lower is treated as incorrect.

`score: null` is handled safely. If the event also carries a `words` array, the
frontend animates those word-level results in sequence instead of waiting only
for the final verdict.

### Deduplication

Duplicate events with the same `attempt_id` are silently ignored.

### Reconnection

On WebSocket close, the app retries with exponential backoff (1 s, 2 s, 4 s, 8 s, 16 s) up to 5 attempts. A yellow "Reconnecting..." banner appears. After max retries, an error state with a Retry button is shown. A disconnection does not clear the last displayed result.

### Unknown events / extra fields

Unknown event types and unknown fields within known events are silently ignored — the page will not crash.

## UI states

```
idle → connecting → ready → listening → processing → evaluated
                                                    → low_confidence → ready
                    error ← disconnected ←──────────┘
```

All states are visually distinct. The connection dot in the top bar changes color (grey = idle, yellow = connecting, green = connected, red = error).

## Phoneme / unit display

The result card renders phoneme-level feedback:

- **English**: each phoneme is shown in a chip. Green = matches expected, red = mismatch.
- **Chinese (Mandarin)**: each syllable unit is shown with segmental + tone annotation. Wrong tones are highlighted in red with `T2→T3` style labels.

The component does not assume units are Latin letters — it renders whatever string the backend provides.

## Exercise configuration

Each exercise carries separate target, sentence, locale, task, and scene fields:

```js
{
  target: 'STAR',
  sentence: 'Twinkle, twinkle, little star',
  language: 'en-US',
  task: 'letter_name_spelling',
  scene: 'meadow'
}
```

`language`, `task`, and `scene` are stored and displayed independently. No "English/Chinese" toggle that loses locale or task.

The current prototype has five hardcoded English exercises. Once the session
creation endpoint is ready, these should come from the backend.

## Real microphone mode

Open `http://localhost:8765`, start a session, then tap the
microphone. The page opens a separate audio WebSocket, requests microphone
permission, resamples mono input to 16 kHz, and sends exact 40 ms PCM s16le
frames. Tap again to finish; trailing silence also finishes after about 1.6
seconds. The evaluated result is accepted from both the audio socket and event
socket and deduplicated by `attempt_id`, so a late duplicate cannot replay the
same level-up animation.

Real sentence scoring requires `PHONETIC_ENABLED=true` on the backend and the
phoneme model already present in its local cache. Without it, the backend can
report audio/DTW telemetry but cannot approve a sentence from positive
full-phrase phonetic evidence. The UI still works, but it falls back to the
non-phonetic result path.

Microphone APIs require a secure context; `localhost` is allowed for local
testing. Production must use HTTPS/WSS.

## Remaining integration work

- Session creation endpoint (`POST /sessions`) — currently mocked
- Browser-specific microphone QA and AudioWorklet migration
- Server-side VAD calibration
- Authentication
- Locale/task selection UI (currently hardcoded to `en-US` + `sentence_pronunciation`)
- Persisting results (no `localStorage` usage, per privacy requirements)

## Privacy

- No audio is recorded, stored, or logged by this frontend
- `session_id` is random — no names, emails, or personal data
- No data is written to `localStorage` or `sessionStorage`
- Console logs contain no personal information
- In production, switch to `wss://` and add authentication

## Remaining product decisions

Key decisions for the next integration pass:

1. Session creation endpoint and payload
2. How to start/stop a pronunciation attempt (command from UI → backend)
3. Agreed `confidence` threshold for low-confidence state
4. Microphone permission denied handling
