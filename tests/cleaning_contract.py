"""Public black-box contract shared by the cleaning tests and benchmark."""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
import os

from app.worker import assemble_word_window


Cleaner = Callable[..., bytes]


def load_cleaner() -> Cleaner:
    """Load ``module:function`` without depending on cleaning internals."""
    entrypoint = os.getenv("CLEANING_ENTRYPOINT", "app.cleaning:clean_pcm")
    try:
        module_name, function_name = entrypoint.split(":", maxsplit=1)
    except ValueError as error:
        raise RuntimeError(
            "CLEANING_ENTRYPOINT must have the form 'module:function'"
        ) from error

    try:
        module = import_module(module_name)
    except ModuleNotFoundError as error:
        raise RuntimeError(
            f"Cleaning not found at {entrypoint!r}. Implement it or set "
            "CLEANING_ENTRYPOINT to the colleague's public function."
        ) from error

    cleaner = getattr(module, function_name, None)
    if not callable(cleaner):
        raise RuntimeError(f"{entrypoint!r} is not callable")
    return cleaner


def clean_redis_records(records: list[dict[bytes, bytes]]) -> bytes:
    """Adapt real Redis stream records to one complete cleaning call."""
    cleaner = load_cleaner()
    return cleaner(assemble_word_window(records))
