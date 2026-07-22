from __future__ import annotations

import json
from pathlib import Path

import pytest

from amie_self_play.config import ModelConfigError, Settings, load_model_catalog
from amie_self_play.prompt_config import (
    PromptConfigError,
    load_default_prompt_catalog,
    load_prompt_catalog,
    save_prompt_config,
    validate_agent_prompt_values,
)
from amie_self_play.prompts import doctor_messages


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


def test_prompt_catalog_applies_partial_agent_override(tmp_path: Path) -> None:
    path = tmp_path / "prompts.toml"
    path.write_text(
        '[agent.doctor]\nsystem = "[ROLE:DOCTOR]\\nCUSTOM_DOCTOR_PROMPT\\n${improvement_context}"\n',
        encoding="utf-8",
    )

    catalog = load_prompt_catalog(path)
    messages = doctor_messages([], [], [], prompts=catalog)

    assert "CUSTOM_DOCTOR_PROMPT" in messages[0]["content"]
    assert "你是一名富有同理心的临床医生" not in messages[0]["content"]


def test_prompt_config_round_trips_through_web_editor_serializer(
    tmp_path: Path,
) -> None:
    defaults = load_default_prompt_catalog()
    path = tmp_path / "prompts.toml"

    save_prompt_config(path, defaults.data)

    assert load_prompt_catalog(path).data == defaults.data


def test_prompt_update_rejects_missing_template_placeholder() -> None:
    defaults = load_default_prompt_catalog()
    doctor = defaults.data["agent"]["doctor"]
    values = dict(doctor)
    values["system"] = "[ROLE:DOCTOR]\n缺少动态改进上下文"

    with pytest.raises(PromptConfigError, match="improvement_context"):
        validate_agent_prompt_values("doctor", values, defaults)


def test_builtin_prompt_defaults_cannot_be_overwritten() -> None:
    defaults = load_default_prompt_catalog()

    with pytest.raises(PromptConfigError, match="不能覆盖"):
        save_prompt_config(defaults.source, defaults.data)
