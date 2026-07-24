"""Public black-box contract shared by the cleaning tests and benchmark."""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
import os


Cleaner = Callable[..., bytes]


def load_cleaner() -> Cleaner:
    """Load ``module:function`` without depending on cleaning internals."""
    entrypoint = os.getenv("CLEANING_ENTRYPOINT", "app.cleaning:clean_audio")
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


def clean_redis_record(record: dict[bytes, bytes]) -> bytes:
    """Adapt one Redis record to the agreed public cleaning call."""
    cleaner = load_cleaner()
    return cleaner(record[b"audio"])
