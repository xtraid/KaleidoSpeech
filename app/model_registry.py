"""Version activation and rollback for decision-engine artifacts."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def load_registry(path: Path) -> dict[str, Any]:
    registry = json.loads(path.read_text(encoding="utf-8"))
    active = registry.get("active")
    models = registry.get("models")
    if not isinstance(models, dict) or active not in models:
        raise ValueError("Registry active model is missing")
    if models[active].get("status") != "approved":
        raise ValueError("Active model must be approved")
    return registry


def activate_model(path: Path, version: str) -> None:
    registry = load_registry(path)
    model = registry["models"].get(version)
    if model is None or model.get("status") != "approved":
        raise ValueError("Only an approved registered model can be activated")
    previous = registry["active"]
    if previous == version:
        return
    registry.setdefault("history", []).append(previous)
    registry["active"] = version
    _atomic_write(path, registry)


def rollback_model(path: Path) -> str:
    registry = load_registry(path)
    history = registry.get("history")
    if not isinstance(history, list) or not history:
        raise ValueError("No previous model version to restore")
    previous = history.pop()
    if registry["models"].get(previous, {}).get("status") != "approved":
        raise ValueError("Previous model is no longer approved")
    registry["active"] = previous
    _atomic_write(path, registry)
    return previous


def _atomic_write(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)
