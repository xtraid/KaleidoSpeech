"""Microphone → Redis → online known-sentence alignment demo."""

from __future__ import annotations

import argparse
import json
import time
import uuid

from app.config import get_settings
from app.redis_bus import client, publish_audio_frame, publish_ui_event
from app.sentence_repository import (
    find_sentence_benchmark,
    save_sentence_attempt,
)
from app.sentence_streaming import SentenceStreamingSession
from app.worker import parse_audio_frame


def main() -> None:
    import sounddevice as sd

    parser = argparse.ArgumentParser()
    parser.add_argument("sentence_id")
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--session-id")
    parser.add_argument("--voice-threshold", type=float, default=0.012)
    args = parser.parse_args()
    benchmark = find_sentence_benchmark(args.sentence_id)
    if benchmark is None:
        parser.error(f"unknown sentence benchmark: {args.sentence_id}")

    settings = get_settings()
    session_id = args.session_id or str(uuid.uuid4())
    inference = SentenceStreamingSession(
        benchmark, voice_threshold=args.voice_threshold
    )
    errors: list[dict] = []
    print(f"Sessione: {session_id}")
    print(f"Pronuncia: {benchmark.text}")
    input("Premi Invio per iniziare...")

    last_id: bytes | str = "0-0"
    frame_count = round(args.duration * 1_000 / settings.frame_ms)
    with sd.RawInputStream(
        samplerate=settings.sample_rate,
        blocksize=settings.samples_per_frame,
        channels=1,
        dtype="int16",
    ) as microphone:
        for sequence in range(frame_count):
            frame, overflow = microphone.read(settings.samples_per_frame)
            if overflow:
                print("ERRORE AUDIO", flush=True)
            publish_audio_frame(
                session_id, sequence, time.monotonic_ns(), bytes(frame)
            )
            batches = client.xread(
                {f"audio:{session_id}": last_id}, count=1, block=1_000
            )
            if not batches:
                continue
            _, messages = batches[0]
            message_id, fields = messages[0]
            last_id = message_id
            _, transported_pcm = parse_audio_frame(fields)
            for event in inference.push_frame(transported_pcm):
                publish_ui_event(session_id, event)
                if event["type"] == "pronunciation.error":
                    errors.append(event)
                    print(
                        f"{event['detected_at_ms']:05d} ms  ERRORE "
                        f"su «{event['unit']}» "
                        f"(iniziato a {event['start_ms']} ms)",
                        flush=True,
                    )
                elif event["type"] == "sentence.alignment":
                    print(
                        f"{event['elapsed_ms']:05d} ms  "
                        f"{event['unit']}  ASCOLTO",
                        flush=True,
                    )

    result = inference.finish()
    result["attempt_id"] = save_sentence_attempt(
        session_id, benchmark.id, result, errors
    )
    publish_ui_event(session_id, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
