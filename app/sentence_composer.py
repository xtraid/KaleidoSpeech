"""Small deterministic English sentence composer for benchmark prompts."""

import random
import re
from dataclasses import dataclass
from hashlib import sha256
from typing import Iterable


COMPOSER_VERSION = "en-templates-v1"
CURRENT_VOCABULARY = frozenset({"bird", "follow", "forward", "happy", "learn"})
_WORD = re.compile(r"^[a-z]+(?:'[a-z]+)?$")

_CURRENT_TEMPLATES = (
    "The happy bird will learn to fly forward and follow the flock.",
    "A happy bird can learn to move forward and follow the flock.",
)
_GENERIC_TEMPLATES = (
    "Please read these words clearly: {targets}.",
    "Now say the following words naturally: {targets}.",
)


@dataclass(frozen=True)
class SentencePrompt:
    text: str
    target_words: tuple[str, ...]
    support_words: tuple[str, ...]
    template_id: str
    version: str = COMPOSER_VERSION


def _normalize_targets(target_words: Iterable[str]) -> tuple[str, ...]:
    normalized = tuple(dict.fromkeys(word.strip().casefold() for word in target_words))
    if not normalized:
        raise ValueError("At least one target word is required")
    if any(not _WORD.fullmatch(word) for word in normalized):
        raise ValueError("Target words must be simple English words")
    return normalized


def compose_sentence(
    target_words: Iterable[str],
    *,
    seed: int | str | None = None,
) -> SentencePrompt:
    """Build a reproducible prompt containing every requested target."""

    targets = _normalize_targets(target_words)
    stable_seed = (
        str(seed)
        if seed is not None
        else sha256("\0".join(targets).encode()).hexdigest()
    )
    randomizer = random.Random(stable_seed)

    if set(targets) == CURRENT_VOCABULARY:
        index = randomizer.randrange(len(_CURRENT_TEMPLATES))
        text = _CURRENT_TEMPLATES[index]
        template_id = f"current-vocabulary-{index + 1}"
    else:
        readable_targets = (
            targets[0]
            if len(targets) == 1
            else ", ".join(targets[:-1]) + f", and {targets[-1]}"
        )
        index = randomizer.randrange(len(_GENERIC_TEMPLATES))
        text = _GENERIC_TEMPLATES[index].format(targets=readable_targets)
        template_id = f"controlled-list-{index + 1}"

    tokens = re.findall(r"[a-z]+(?:'[a-z]+)?", text.casefold())
    target_set = set(targets)
    remaining = [token for token in tokens if token not in target_set]

    return SentencePrompt(
        text=text,
        target_words=targets,
        support_words=tuple(dict.fromkeys(remaining)),
        template_id=template_id,
    )
