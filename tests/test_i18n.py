import pytest

from app.i18n import message


def test_feedback_is_available_in_english_and_italian() -> None:
    assert message("en-US", "retry").startswith("Insufficient")
    assert message("it-IT", "retry").startswith("Audio insufficiente")


def test_unsupported_feedback_locale_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported locale"):
        message("fr-FR", "retry")
