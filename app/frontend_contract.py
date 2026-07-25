"""Adapters from internal decisions to the stable browser event contract."""

from __future__ import annotations

from typing import Any


def pronunciation_evaluated_event(
    *,
    attempt_id: str,
    session_id: str,
    result: dict[str, Any],
) -> dict[str, Any]:
    phones = result.get("phones") or []
    status = str(result.get("status", "UNDECIDABLE"))
    if status == "CORRECT":
        score: float | None = 1.0
    elif status == "INCORRECT":
        score = 0.0
    else:
        score = None
    versions = result.get("versions") or {}
    confidence = result.get("word_identity_confidence")
    if not isinstance(confidence, (int, float)):
        confidence = 0.0
    return {
        "type": "pronunciation.evaluated",
        "version": 1,
        "attempt_id": attempt_id,
        "session_id": session_id,
        "phase": "evaluated",
        "status": status,
        "reason_codes": result.get("reason_codes", []),
        "expected_word": result.get("target_word"),
        "recognized_word": result.get("recognized_word"),
        "expected_text": result.get("expected_text"),
        "mode": result.get("mode", "word_pronunciation"),
        "words": result.get("words", []),
        "expected_phonemes": [
            phone.get("expected") for phone in phones if phone.get("expected")
        ],
        "detected_phonemes": [
            phone.get("observed") or "?" for phone in phones if phone.get("expected")
        ],
        "confidence": max(0.0, min(1.0, float(confidence))),
        "score": score,
        "benchmark_version": versions.get("lexicon"),
        "engine_version": versions.get("decision"),
        "result": result,
    }
