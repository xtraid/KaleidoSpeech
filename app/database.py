"""Safe, short-lived SQLite connections."""

import sqlite3
from collections.abc import Generator
from contextlib import contextmanager

from app.config import get_settings


@contextmanager
def db_connection() -> Generator[sqlite3.Connection, None, None]:
    """Open one SQLite connection and always close it."""
    database_path = get_settings().sqlite_path
    database_path.parent.mkdir(parents=True, exist_ok=True)

    database = sqlite3.connect(
        database_path,
        timeout=5.0,
        isolation_level=None,
    )
    database.row_factory = sqlite3.Row
    database.execute("PRAGMA foreign_keys = ON")
    database.execute("PRAGMA busy_timeout = 5000")
    try:
        yield database
    finally:
        database.close()


# Backward-compatible name for code written before the implementation guide.
connection = db_connection
