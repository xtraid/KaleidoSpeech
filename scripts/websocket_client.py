"""Small reference client for a pre-recorded 16 kHz PCM file."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import websockets


async def stream(
    url: str,
    target: str,
    expected_text: str | None,
    pcm_path: Path,
    token: str | None,
) -> None:
    headers = {"Authorization": f"Bearer {token}"} if token else None
    async with websockets.connect(url, additional_headers=headers) as socket:
        command = {"type": "session.start", "target_words": [target]}
        if expected_text:
            command["expected_text"] = expected_text
        await socket.send(json.dumps(command))
        print(await socket.recv())
        pcm = pcm_path.read_bytes()
        for offset in range(0, len(pcm), 1_280):
            frame = pcm[offset : offset + 1_280]
            if len(frame) != 1_280:
                break
            await socket.send(frame)
            event = json.loads(await socket.recv())
            if event.get("word_updates"):
                print(json.dumps({"word_updates": event["word_updates"]}))
            await asyncio.sleep(0.04)
        await socket.send(json.dumps({"type": "session.finish"}))
        print(await socket.recv())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pcm", type=Path)
    parser.add_argument("--target", required=True)
    parser.add_argument("--expected-text")
    parser.add_argument(
        "--url", default="ws://127.0.0.1:8000/streaming/sessions/demo"
    )
    parser.add_argument("--token")
    arguments = parser.parse_args()
    asyncio.run(stream(
        arguments.url,
        arguments.target,
        arguments.expected_text,
        arguments.pcm,
        arguments.token,
    ))


if __name__ == "__main__":
    main()
