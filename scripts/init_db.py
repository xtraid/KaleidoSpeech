"""Initialize the SQLite schema."""

from app.database import connection


def main() -> None:
    with connection() as database:
        database.execute(
            """
            CREATE TABLE IF NOT EXISTS benchmarks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                score REAL NOT NULL CHECK (score >= 0 AND score <= 1),
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )


if __name__ == "__main__":
    main()

