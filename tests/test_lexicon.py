import pytest

from app.lexicon import EN_US_LEXICON, lookup


def test_lexicon_preserves_stress_and_compatibility_view() -> None:
    variants = lookup("Bird")

    assert variants[0].stress == (0, 1, 0)
    assert EN_US_LEXICON["bird"][0] == list(variants[0].phones)


def test_unknown_lexicon_word_is_explicit() -> None:
    with pytest.raises(LookupError):
        lookup("unknown")
