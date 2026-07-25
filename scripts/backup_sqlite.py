"""Create a consistent SQLite backup without copying a live WAL database."""

from __future__ import annotations

import argparse
from pathlib import Path
import sqlite3

from app.config import get_settings


def backup(destination: Path) -> None:
    source = get_settings().sqlite_path.resolve()
    destination = destination.resolve()
    if source == destination:
        raise ValueError("Backup destination must differ from the source")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(source) as source_db, sqlite3.connect(destination) as target_db:
        source_db.backup(target_db)
        if target_db.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise RuntimeError("Backup integrity check failed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("destination", type=Path)
    backup(parser.parse_args().destination)


if __name__ == "__main__":
    main()
