"""Small versioned message catalog for API-facing feedback."""

from __future__ import annotations


SUPPORTED_LOCALES = frozenset({"en", "it"})
MESSAGES = {
    "en": {
        "retry": "Insufficient audio: please try again.",
        "review": "Recording captured; professional review is required.",
    },
    "it": {
        "retry": "Audio insufficiente: ripetere.",
        "review": "Registrazione acquisita; revisione del logopedista richiesta.",
    },
}


def normalize_locale(locale: str | None) -> str:
    language = (locale or "en").split("-", 1)[0].casefold()
    if language not in SUPPORTED_LOCALES:
        raise ValueError(f"Unsupported locale: {locale}")
    return language


def message(locale: str, key: str) -> str:
    return MESSAGES[normalize_locale(locale)][key]
