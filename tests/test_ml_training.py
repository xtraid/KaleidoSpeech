import pytest

from app.ml_features import FEATURE_NAMES
from app.ml_training import TrainingExample, validate_training_examples


def _example(index: int, split: str) -> TrainingExample:
    return TrainingExample(
        attempt_id=f"a{index}",
        speaker_hash=f"s{index}",
        split=split,
        features=tuple(0.0 for _ in FEATURE_NAMES),
        label="CORRECT",
        rule_based_label="CORRECT",
    )


def test_training_refuses_less_than_required_reviewed_data() -> None:
    with pytest.raises(ValueError, match="At least 500"):
        validate_training_examples([_example(1, "train")])


def test_training_refuses_speaker_leakage() -> None:
    examples = [
        _example(index, ("train", "validation", "test")[index % 3])
        for index in range(500)
    ]
    examples[1] = TrainingExample(
        **{**examples[1].__dict__, "speaker_hash": examples[0].speaker_hash}
    )
    with pytest.raises(ValueError, match="multiple"):
        validate_training_examples(examples)
