"""SQLite connection helpers."""

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager

from app.config import get_settings


@contextmanager
def connection() -> Iterator[sqlite3.Connection]:
    database_path = get_settings().database_path
    database_path.parent.mkdir(parents=True, exist_ok=True)
    database = sqlite3.connect(database_path)
    database.row_factory = sqlite3.Row
    try:
        yield database
        database.commit()
    finally:
        database.close()

