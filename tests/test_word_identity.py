from itertools import permutations

from app.lexicon import EN_US_LEXICON
from app.phonetic_pipeline import DtwWordIdentityGate
from app.word_identity_calibration import (
    LabeledWordDistances,
    calibrate_mismatch_margin,
)


def test_all_twenty_ordered_word_mismatches_are_rejected() -> None:
    words = tuple(EN_US_LEXICON)
    assert len(words) == 5
    pairs = tuple(permutations(words, 2))
    assert len(pairs) == 20

    for target, spoken in pairs:
        distances = {word: 1.0 for word in words}
        distances[spoken] = 0.1
        distances[target] = 0.8
        result = DtwWordIdentityGate(lambda _: distances).evaluate(b"pcm", target)
        assert result.status == "MISMATCH", (target, spoken)
        assert result.recognized_word == spoken


def test_mismatch_margin_is_calibrated_to_far_budget() -> None:
    samples = [
        LabeledWordDistances("bird", "happy", {"bird": 0.2, "happy": 0.3}, "s1"),
        LabeledWordDistances("bird", "happy", {"bird": 0.8, "happy": 0.1}, "s2"),
        LabeledWordDistances("bird", "bird", {"bird": 0.1, "happy": 0.5}, "s3"),
    ]
    calibration = calibrate_mismatch_margin(samples)

    assert calibration.mismatch_margin_threshold > 0.1
    assert calibration.wrong_word_false_acceptance_rate == 0
