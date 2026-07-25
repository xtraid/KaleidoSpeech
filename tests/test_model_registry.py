import json

import pytest

from app.model_registry import activate_model, load_registry, rollback_model


def test_model_activation_and_rollback_are_versioned(tmp_path) -> None:
    path = tmp_path / "registry.json"
    path.write_text(
        json.dumps(
            {
                "active": "rules-v1",
                "models": {
                    "rules-v1": {"status": "approved"},
                    "forest-v1": {"status": "approved"},
                },
                "history": [],
            }
        )
    )

    activate_model(path, "forest-v1")
    assert load_registry(path)["active"] == "forest-v1"
    assert rollback_model(path) == "rules-v1"
    assert load_registry(path)["active"] == "rules-v1"


def test_unapproved_model_cannot_be_activated(tmp_path) -> None:
    path = tmp_path / "registry.json"
    path.write_text(
        json.dumps(
            {
                "active": "rules-v1",
                "models": {
                    "rules-v1": {"status": "approved"},
                    "forest-v1": {"status": "candidate"},
                },
            }
        )
    )
    with pytest.raises(ValueError, match="approved"):
        activate_model(path, "forest-v1")
