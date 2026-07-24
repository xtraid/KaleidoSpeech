"""Redis Stream consumer for pronunciation analysis."""

from app import benchmark_repository, pronunciation_engine


def process_audio(audio: bytes) -> int:
    result = benchmark_repository.save(pronunciation_engine.score(audio))
    return result.id

