from __future__ import annotations

import json
from pathlib import Path

import pytest

from amie_self_play.config import ModelConfigError, Settings, load_model_catalog


def write_config(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def test_catalog_loads_default_and_runtime_settings(tmp_path: Path) -> None:
    path = tmp_path / "models.json"
    write_config(
        path,
        {
            "default_model": "model-beta",
            "timeout_seconds": 45,
            "max_retries": 2,
            "models": [
                {
                    "name": "model-alpha",
                    "endpoint": "https://example.test/chat",
                },
                {
                    "name": "model-beta",
                    "endpoint": "http://127.0.0.1:11434/v1/chat/completions",
                    "model": "upstream-beta",
                },
            ],
        },
    )

    catalog = load_model_catalog(path, Settings())

    assert catalog.default_model == "model-beta"
    assert catalog.timeout_seconds == 45
    assert catalog.max_retries == 2
    assert catalog.by_name("model-beta").api_model == "upstream-beta"


def test_catalog_rejects_unknown_default_model(tmp_path: Path) -> None:
    path = tmp_path / "models.json"
    write_config(
        path,
        {
            "default_model": "missing",
            "models": [
                {"name": "model-alpha", "endpoint": "https://example.test/chat"}
            ],
        },
    )

    with pytest.raises(ModelConfigError, match="default_model"):
        load_model_catalog(path)
