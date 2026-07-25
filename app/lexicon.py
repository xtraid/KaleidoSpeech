"""Versioned MVP lexicon in the IPA inventory emitted by eSpeak."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

EN_US_LEXICON_VERSION = "en-us-espeak-v1"


Stress = Literal[0, 1, 2]


@dataclass(frozen=True)
class PronunciationVariant:
    phones: tuple[str, ...]
    stress: tuple[Stress, ...]

    def __post_init__(self) -> None:
        if len(self.phones) != len(self.stress):
            raise ValueError("Stress must align one-to-one with phones")


# Stress: 0=none, 1=primary, 2=secondary. Regional variants are explicit.
EN_US_PRONUNCIATIONS: dict[str, tuple[PronunciationVariant, ...]] = {
    "bird": (
        PronunciationVariant(("b", "ɜː", "d"), (0, 1, 0)),
        PronunciationVariant(("b", "ɝ", "d"), (0, 1, 0)),
    ),
    "follow": (
        PronunciationVariant(("f", "ɑː", "l", "oʊ"), (0, 1, 0, 0)),
        PronunciationVariant(("f", "ɒ", "l", "oʊ"), (0, 1, 0, 0)),
    ),
    "forward": (
        PronunciationVariant(("f", "ɔːɹ", "w", "ɚ", "d"), (0, 1, 0, 0, 0)),
    ),
    "happy": (
        PronunciationVariant(("h", "æ", "p", "i"), (0, 1, 0, 0)),
    ),
    "learn": (
        PronunciationVariant(("l", "ɜː", "n"), (0, 1, 0)),
        PronunciationVariant(("l", "ɝ", "n"), (0, 1, 0)),
    ),
}

# Compatibility view used by the acoustic pipeline.
EN_US_LEXICON: dict[str, list[list[str]]] = {
    word: [list(variant.phones) for variant in variants]
    for word, variants in EN_US_PRONUNCIATIONS.items()
}

EN_US_SENTENCE_LEXICON: dict[str, list[list[str]]] = {
    **EN_US_LEXICON,
    "a": [["ə"]],
    "can": [["k", "æ", "n"]],
    "fly": [["f", "l", "aɪ"]],
    "the": [["ð", "ə"]],
    "little": [["l", "ɪ", "t", "əl"]],
    "take": [["t", "eɪ", "k"]],
    "one": [["w", "ʌ", "n"]],
    "step": [["s", "t", "ɛ", "p"]],
    "we": [["w", "iː"]],
    "new": [["n", "uː"]],
    "word": [["w", "ɜː", "d"]],
}


def lookup(word: str) -> tuple[PronunciationVariant, ...]:
    try:
        return EN_US_PRONUNCIATIONS[word.strip().casefold()]
    except KeyError as error:
        raise LookupError(f"Word is absent from {EN_US_LEXICON_VERSION}") from error
