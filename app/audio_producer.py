"""Capture canonical microphone frames and publish them outside the callback."""

from __future__ import annotations

import logging
import queue
import threading
import time
import uuid

import numpy as np

from app.config import get_settings
from app.redis_bus import publish_audio_frame


def split_pcm(pcm: bytes, samples_per_frame: int) -> list[bytes]:
    """Split mono int16 PCM into complete frames, dropping a partial tail."""
    bytes_per_frame = samples_per_frame * 2
    return [
        pcm[offset : offset + bytes_per_frame]
        for offset in range(0, len(pcm), bytes_per_frame)
        if len(pcm[offset : offset + bytes_per_frame]) == bytes_per_frame
    ]


class MicrophoneProducer:
    def __init__(self, session_id: str | None = None) -> None:
        self.session_id = session_id or str(uuid.uuid4())
        self.frames: queue.Queue[tuple[int, bytes]] = queue.Queue(maxsize=100)
        self.stop_event = threading.Event()
        self.sequence = 0
        self.dropped_frames = 0

    def _callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: object,
        status: object,
    ) -> None:
        del time_info
        settings = get_settings()
        if status:
            # Production telemetry may record the flag, never the audio.
            pass
        if frames != settings.samples_per_frame:
            return

        item = (time.monotonic_ns(), indata.copy().tobytes())
        try:
            self.frames.put_nowait(item)
        except queue.Full:
            self.dropped_frames += 1

    def _publisher_loop(self) -> None:
        while not self.stop_event.is_set() or not self.frames.empty():
            try:
                captured_at_ns, pcm = self.frames.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                publish_audio_frame(
                    self.session_id,
                    self.sequence,
                    captured_at_ns,
                    pcm,
                )
                self.sequence += 1
            finally:
                self.frames.task_done()

    def run(self) -> None:
        import sounddevice as sd

        settings = get_settings()
        publisher = threading.Thread(
            target=self._publisher_loop,
            name="redis-publisher",
            daemon=True,
        )
        publisher.start()

        try:
            with sd.InputStream(
                samplerate=settings.sample_rate,
                channels=settings.channels,
                dtype=settings.dtype,
                blocksize=settings.samples_per_frame,
                callback=self._callback,
            ):
                logging.getLogger(__name__).info(
                    "audio_session_started session_id=%s", self.session_id
                )
                input()
        finally:
            self.stop_event.set()
            publisher.join(timeout=3)


if __name__ == "__main__":
    MicrophoneProducer().run()
