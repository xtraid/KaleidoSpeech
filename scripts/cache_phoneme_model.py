"""Download the pinned phoneme model into an explicit deployment cache."""

from __future__ import annotations

import argparse
from pathlib import Path

from app.pronunciation_engine import DEFAULT_PHONEME_MODEL, DEFAULT_PHONEME_REVISION


def cache_model(cache_dir: Path) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise RuntimeError("Install with: uv sync --extra phonetic") from error
    cache_dir = cache_dir.resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    downloaded = snapshot_download(
        repo_id=DEFAULT_PHONEME_MODEL,
        revision=DEFAULT_PHONEME_REVISION,
        cache_dir=cache_dir,
    )
    return Path(downloaded)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("data/model-cache")
    )
    print(cache_model(parser.parse_args().cache_dir))


if __name__ == "__main__":
    main()
