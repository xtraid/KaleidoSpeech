"""Persistence boundary for pronunciation benchmark results."""

from dataclasses import dataclass

from app.database import connection


@dataclass(frozen=True)
class Benchmark:
    id: int
    score: float
    created_at: str


def save(score: float) -> Benchmark:
    with connection() as database:
        cursor = database.execute(
            "INSERT INTO benchmarks (score) VALUES (?) RETURNING id, score, created_at",
            (score,),
        )
        row = cursor.fetchone()
    if row is None:
        raise RuntimeError("Benchmark insertion returned no row")
    return Benchmark(id=row["id"], score=row["score"], created_at=row["created_at"])

