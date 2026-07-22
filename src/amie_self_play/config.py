from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_DIR / "config" / "model_apis.json"
DEFAULT_PROMPT_CONFIG_PATH = PROJECT_DIR / "config" / "prompts.toml"
_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


class ModelConfigError(ValueError):
    """Raised when the model API configuration is missing or invalid."""


@dataclass(frozen=True, slots=True)
class Settings:
    """Runtime settings shared by the model catalog and legacy callers.

    ``endpoint`` and ``model`` remain available for callers that construct a
    single legacy client directly. Normal application startup uses
    ``config_path`` and the catalog loaded from that file.
    """

    endpoint: str | None = None
    model: str | None = None
    timeout_seconds: float = 120.0
    max_retries: int = 1
    config_path: Path = DEFAULT_CONFIG_PATH
    prompt_config_path: Path = DEFAULT_PROMPT_CONFIG_PATH
    prompt_admin_token: str | None = None

    @classmethod
    def from_env(cls) -> "Settings":
        config_value = os.getenv("AMIE_MODEL_CONFIG", "").strip()
        if not config_value:
            config_value = os.getenv("MODEL_API_CONFIG", "").strip()
        prompt_config_value = os.getenv("AMIE_PROMPT_CONFIG", "").strip()
        return cls(
            timeout_seconds=float(os.getenv("AMIE_TIMEOUT", "120")),
            max_retries=int(os.getenv("AMIE_MAX_RETRIES", "1")),
            config_path=Path(config_value).expanduser() if config_value else DEFAULT_CONFIG_PATH,
            prompt_config_path=(
                Path(prompt_config_value).expanduser()
                if prompt_config_value
                else DEFAULT_PROMPT_CONFIG_PATH
            ),
            prompt_admin_token=os.getenv("AMIE_PROMPT_ADMIN_TOKEN", "").strip()
            or None,
        )


@dataclass(frozen=True, slots=True)
class ModelAPI:
    """One selectable model and the HTTP contract used to call it."""

    name: str
    endpoint: str
    api_model: str
    real_name: str
    model_field: str = "model"
    api_key_env: str | None = None
    api_key_header: str = "Authorization"
    api_key_prefix: str = "Bearer "
    headers: dict[str, str] = field(default_factory=dict)
    body: dict[str, Any] = field(default_factory=dict)
    context_length: int = 0
    support_stream: bool = False
    image_input: bool = False

    @classmethod
    def from_dict(cls, raw: object, *, index: int) -> "ModelAPI":
        if not isinstance(raw, dict):
            raise ModelConfigError(f"models[{index}] 必须是 JSON object")

        def required_string(key: str) -> str:
            value = raw.get(key)
            if not isinstance(value, str) or not value.strip():
                raise ModelConfigError(f"models[{index}].{key} 必须是非空字符串")
            return value.strip()

        def optional_string(key: str, default: str | None = None) -> str | None:
            value = raw.get(key, default)
            if value is None:
                return None
            if not isinstance(value, str):
                raise ModelConfigError(f"models[{index}].{key} 必须是字符串")
            value = value.strip()
            return value or None

        name = required_string("name")
        endpoint = required_string("endpoint")
        if not endpoint.startswith(("http://", "https://")):
            raise ModelConfigError(f"models[{index}].endpoint 必须以 http:// 或 https:// 开头")
        api_model = optional_string("model") or name
        real_name = optional_string("real_name") or name
        model_field = optional_string("model_field") or "model"

        headers = raw.get("headers", {})
        if not isinstance(headers, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in headers.items()
        ):
            raise ModelConfigError(f"models[{index}].headers 必须是字符串到字符串的 object")
        body = raw.get("body", {})
        if not isinstance(body, dict):
            raise ModelConfigError(f"models[{index}].body 必须是 object")

        context_length = raw.get("context_length", 0)
        if not isinstance(context_length, int) or context_length < 0:
            raise ModelConfigError(f"models[{index}].context_length 必须是非负整数")
        for key in ("support_stream", "image_input"):
            if key in raw and not isinstance(raw[key], bool):
                raise ModelConfigError(f"models[{index}].{key} 必须是 boolean")

        api_key_env = optional_string("api_key_env")
        api_key_header = optional_string("api_key_header") or "Authorization"
        api_key_prefix = raw.get("api_key_prefix", "Bearer ")
        if not isinstance(api_key_prefix, str):
            raise ModelConfigError(f"models[{index}].api_key_prefix 必须是字符串")

        return cls(
            name=name,
            endpoint=endpoint,
            api_model=api_model,
            real_name=real_name,
            model_field=model_field,
            api_key_env=api_key_env,
            api_key_header=api_key_header,
            api_key_prefix=api_key_prefix,
            headers=dict(headers),
            body=dict(body),
            context_length=context_length,
            support_stream=bool(raw.get("support_stream", False)),
            image_input=bool(raw.get("image_input", False)),
        )

    def public_metadata(self) -> dict[str, Any]:
        """Return only fields safe to expose to the browser."""

        return {
            "name": self.name,
            "real_name": self.real_name,
            "context_length": self.context_length,
            "support_stream": self.support_stream,
            "image_input": self.image_input,
        }


@dataclass(frozen=True, slots=True)
class ModelCatalog:
    default_model: str
    models: tuple[ModelAPI, ...]
    timeout_seconds: float = 120.0
    max_retries: int = 1

    def by_name(self, name: str) -> ModelAPI:
        for model in self.models:
            if model.name == name:
                return model
        raise ModelConfigError(f"模型 {name!r} 不在配置文件的 models 列表中")

    def public_models(self) -> list[dict[str, Any]]:
        return [model.public_metadata() for model in self.models]


def load_model_catalog(path: Path, settings: Settings | None = None) -> ModelCatalog:
    """Load and validate the user-provided model API configuration."""

    if not path.is_file():
        raise ModelConfigError(
            f"找不到模型 API 配置文件：{path}。请复制 config/model_apis.example.json "
            "为 config/model_apis.json 后再启动服务。"
        )
    try:
        with path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ModelConfigError(f"模型 API 配置不是有效 JSON：{path}: {exc}") from exc
    except OSError as exc:
        raise ModelConfigError(f"无法读取模型 API 配置：{path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ModelConfigError("模型 API 配置顶层必须是 JSON object")
    entries = raw.get("models")
    if not isinstance(entries, list) or not entries:
        raise ModelConfigError("模型 API 配置必须包含非空 models 数组")
    models = tuple(ModelAPI.from_dict(item, index=index) for index, item in enumerate(entries))
    names = [model.name for model in models]
    if len(set(names)) != len(names):
        raise ModelConfigError("models.name 必须唯一")

    fallback = (settings.model if settings else None) or names[0]
    default_model = raw.get("default_model", fallback)
    if not isinstance(default_model, str) or default_model not in names:
        raise ModelConfigError("default_model 必须匹配 models 中的 name")

    timeout_seconds = raw.get("timeout_seconds", settings.timeout_seconds if settings else 120.0)
    max_retries = raw.get("max_retries", settings.max_retries if settings else 1)
    if not isinstance(timeout_seconds, (int, float)) or timeout_seconds <= 0:
        raise ModelConfigError("timeout_seconds 必须是正数")
    if not isinstance(max_retries, int) or max_retries < 0:
        raise ModelConfigError("max_retries 必须是非负整数")
    return ModelCatalog(
        default_model=default_model,
        models=models,
        timeout_seconds=float(timeout_seconds),
        max_retries=max_retries,
    )


def expand_environment(value: str) -> str:
    """Expand ${ENV_VAR} references used by configurable headers."""

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in os.environ:
            raise ModelConfigError(f"环境变量 {name} 未设置")
        return os.environ[name]

    return _ENV_PATTERN.sub(replace, value)
