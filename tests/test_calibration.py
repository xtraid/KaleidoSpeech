import pytest

from app.calibration import LabeledPhoneScore, calibrate_phone_thresholds


def test_calibration_finds_separating_phone_threshold() -> None:
    samples = [
        LabeledPhoneScore("b", -2.0, True, "s1"),
        LabeledPhoneScore("b", -1.0, True, "s2"),
        LabeledPhoneScore("b", 1.0, False, "s3"),
        LabeledPhoneScore("b", 2.0, False, "s4"),
    ]
    result = calibrate_phone_thresholds(samples)["b"]

    assert -1.0 <= result.threshold < 1.0
    assert result.false_acceptance_rate == 0
    assert result.false_rejection_rate == 0


def test_calibration_refuses_unmeetable_error_budget() -> None:
    samples = [
        LabeledPhoneScore("b", 1.0, True, "s1"),
        LabeledPhoneScore("b", -1.0, False, "s2"),
    ]
    with pytest.raises(ValueError, match="No threshold"):
        calibrate_phone_thresholds(samples)
