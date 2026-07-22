from __future__ import annotations

import copy
import json
import os
import string
import tempfile
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import DEFAULT_PROMPT_CONFIG_PATH


class PromptConfigError(ValueError):
    """Raised when a prompt configuration cannot be loaded or rendered."""


AGENT_METADATA = (
    ("vignette", "Vignette", "根据疾病生成结构化模拟病例"),
    ("patient", "Patient", "根据病例扮演在线问诊患者"),
    ("doctor", "Doctor", "执行问诊并吸收历史 Critic 反馈"),
    ("moderator", "Moderator", "判断当前医患对话是否可以结束"),
    ("ddx", "Doctor DDx", "根据本轮对话整理鉴别诊断"),
    ("critic", "Critic", "复盘医生表现并提出改进建议"),
    ("accuracy_rater", "Accuracy Rater", "评审鉴别诊断的语义匹配准确性"),
    ("patient_actor_rater", "Patient Actor Rater", "按患者体验量表进行代理评分"),
    ("specialist_rater", "Specialist Rater", "按专科、PACES 和管理量表评分"),
    ("auto_paces_rater", "Auto PACES Rater", "基于对话进行自动 PACES 评分"),
    ("json_repair", "JSON Repair", "修复不符合结构要求的模型输出"),
)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(result.get(key), dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _read_toml(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as handle:
            data = tomllib.load(handle)
    except tomllib.TOMLDecodeError as exc:
        raise PromptConfigError(f"提示词配置不是有效 TOML：{path}: {exc}") from exc
    except OSError as exc:
        raise PromptConfigError(f"无法读取提示词配置：{path}: {exc}") from exc
    if not isinstance(data, dict):
        raise PromptConfigError(f"提示词配置顶层必须是 object：{path}")
    return data


def default_prompt_config_path() -> Path:
    return DEFAULT_PROMPT_CONFIG_PATH.with_name("prompts.example.toml")


@dataclass(frozen=True, slots=True)
class PromptCatalog:
    data: dict[str, Any]
    source: Path

    def _get(self, section: str, key: str) -> str:
        container: object = self.data
        for part in section.split("."):
            if not isinstance(container, dict):
                container = None
                break
            container = container.get(part)
        value = container.get(key) if isinstance(container, dict) else None
        if not isinstance(value, str):
            raise PromptConfigError(
                f"提示词配置缺少字符串字段 [{section}] {key}（来源：{self.source}）"
            )
        return value

    def render(self, agent: str, part: str, **values: object) -> str:
        template = self._get(f"agent.{agent}", part)
        try:
            return string.Template(template).substitute(
                {key: str(value) for key, value in values.items()}
            )
        except (KeyError, ValueError) as exc:
            raise PromptConfigError(
                f"提示词模板 agent.{agent}.{part} 的变量无法替换：{exc}"
            ) from exc

    def schema(self, name: str, **values: object) -> str:
        template = self._get("schema", name)
        try:
            return string.Template(template).substitute(
                {key: str(value) for key, value in values.items()}
            )
        except (KeyError, ValueError) as exc:
            raise PromptConfigError(f"提示词 schema {name} 的变量无法替换：{exc}") from exc

    def example(self, name: str) -> str:
        return self._get("example", name)

    def meta(self, name: str, **values: object) -> str:
        template = self._get("meta", name)
        try:
            return string.Template(template).substitute(
                {key: str(value) for key, value in values.items()}
            )
        except (KeyError, ValueError) as exc:
            raise PromptConfigError(f"提示词 meta {name} 的变量无法替换：{exc}") from exc

    @property
    def doctor_opening(self) -> str:
        return self._get("meta", "doctor_opening")

    @property
    def cli_system(self) -> str:
        return self._get("meta", "cli_system")


def load_prompt_catalog(path: Path | None = None) -> PromptCatalog:
    """Load the example prompts and apply an optional user override file."""

    env_path = os.getenv("AMIE_PROMPT_CONFIG", "").strip()
    selected = path or (
        Path(env_path).expanduser() if env_path else DEFAULT_PROMPT_CONFIG_PATH
    )
    example_path = default_prompt_config_path()
    if not example_path.is_file():
        raise PromptConfigError(f"找不到内置提示词示例：{example_path}")
    data = _read_toml(example_path)
    if selected.is_file() and selected.resolve() != example_path.resolve():
        data = _deep_merge(data, _read_toml(selected))
        source = selected
    elif selected.is_file():
        source = selected
    elif selected.resolve() == DEFAULT_PROMPT_CONFIG_PATH.resolve():
        source = example_path
    else:
        raise PromptConfigError(f"找不到提示词配置文件：{selected}")
    return PromptCatalog(data=data, source=source)


def load_default_prompt_catalog() -> PromptCatalog:
    path = default_prompt_config_path()
    if not path.is_file():
        raise PromptConfigError(f"找不到内置提示词示例：{path}")
    return PromptCatalog(data=_read_toml(path), source=path)


def _template_identifiers(template: str, *, field: str) -> set[str]:
    parsed = string.Template(template)
    if not parsed.is_valid():
        raise PromptConfigError(f"提示词字段 {field} 包含无效的 $ 模板语法")
    return set(parsed.get_identifiers())


def validate_agent_prompt_values(
    agent: str,
    values: object,
    defaults: PromptCatalog,
) -> dict[str, str]:
    default_agents = defaults.data.get("agent")
    default_section = (
        default_agents.get(agent) if isinstance(default_agents, dict) else None
    )
    if not isinstance(default_section, dict):
        raise PromptConfigError(f"未知 agent：{agent}")
    if not isinstance(values, dict):
        raise PromptConfigError("values 必须是字段名到提示词文本的 object")

    expected = set(default_section)
    supplied = set(values)
    missing = sorted(expected - supplied)
    extra = sorted(supplied - expected)
    if missing or extra:
        details: list[str] = []
        if missing:
            details.append(f"缺少字段：{', '.join(missing)}")
        if extra:
            details.append(f"未知字段：{', '.join(extra)}")
        raise PromptConfigError("；".join(details))

    validated: dict[str, str] = {}
    for key in default_section:
        value = values[key]
        field = f"agent.{agent}.{key}"
        if not isinstance(value, str):
            raise PromptConfigError(f"提示词字段 {field} 必须是字符串")
        if not value.strip():
            raise PromptConfigError(f"提示词字段 {field} 不能为空")
        if len(value) > 100_000:
            raise PromptConfigError(f"提示词字段 {field} 不能超过 100000 个字符")
        required = _template_identifiers(str(default_section[key]), field=field)
        actual = _template_identifiers(value, field=field)
        missing_placeholders = sorted(required - actual)
        unknown_placeholders = sorted(actual - required)
        if missing_placeholders:
            names = ", ".join(f"${{{name}}}" for name in missing_placeholders)
            raise PromptConfigError(f"提示词字段 {field} 缺少必要变量：{names}")
        if unknown_placeholders:
            names = ", ".join(f"${{{name}}}" for name in unknown_placeholders)
            raise PromptConfigError(f"提示词字段 {field} 包含未知变量：{names}")
        validated[key] = value
    return validated


def update_agent_prompt_data(
    catalog: PromptCatalog,
    defaults: PromptCatalog,
    agent: str,
    values: object,
) -> dict[str, Any]:
    validated = validate_agent_prompt_values(agent, values, defaults)
    data = copy.deepcopy(catalog.data)
    agents = data.get("agent")
    if not isinstance(agents, dict):
        raise PromptConfigError("提示词配置缺少 [agent] sections")
    agents[agent] = validated
    return data


def reset_agent_prompt_data(
    catalog: PromptCatalog,
    defaults: PromptCatalog,
    agent: str,
) -> dict[str, Any]:
    default_agents = defaults.data.get("agent")
    default_section = (
        default_agents.get(agent) if isinstance(default_agents, dict) else None
    )
    if not isinstance(default_section, dict):
        raise PromptConfigError(f"未知 agent：{agent}")
    return update_agent_prompt_data(catalog, defaults, agent, default_section)


def agent_prompt_payload(
    catalog: PromptCatalog, defaults: PromptCatalog
) -> list[dict[str, Any]]:
    current_agents = catalog.data.get("agent")
    default_agents = defaults.data.get("agent")
    if not isinstance(current_agents, dict) or not isinstance(default_agents, dict):
        raise PromptConfigError("提示词配置缺少 [agent] sections")

    result: list[dict[str, Any]] = []
    for name, label, description in AGENT_METADATA:
        current = current_agents.get(name)
        default = default_agents.get(name)
        if not isinstance(current, dict) or not isinstance(default, dict):
            raise PromptConfigError(f"提示词配置缺少 [agent.{name}]")
        fields = []
        for field_name, default_value in default.items():
            current_value = current.get(field_name)
            if not isinstance(current_value, str) or not isinstance(default_value, str):
                raise PromptConfigError(
                    f"提示词字段 agent.{name}.{field_name} 必须是字符串"
                )
            fields.append(
                {
                    "name": field_name,
                    "value": current_value,
                    "default": default_value,
                    "required_placeholders": sorted(
                        _template_identifiers(
                            default_value, field=f"agent.{name}.{field_name}"
                        )
                    ),
                }
            )
        result.append(
            {
                "name": name,
                "label": label,
                "description": description,
                "modified": current != default,
                "fields": fields,
            }
        )
    return result


def _toml_key(value: str) -> str:
    if value.replace("_", "").replace("-", "").isalnum():
        return value
    return json.dumps(value, ensure_ascii=False)


def _append_toml_tables(
    lines: list[str], prefix: tuple[str, ...], section: Mapping[str, Any]
) -> None:
    strings = [
        (key, value) for key, value in section.items() if isinstance(value, str)
    ]
    children = [
        (key, value) for key, value in section.items() if isinstance(value, dict)
    ]
    unsupported = [
        key for key, value in section.items() if not isinstance(value, (str, dict))
    ]
    if unsupported:
        raise PromptConfigError(
            f"提示词配置包含不支持的字段类型：{', '.join(unsupported)}"
        )

    if strings:
        if lines and lines[-1] != "":
            lines.append("")
        dotted = ".".join(_toml_key(part) for part in prefix)
        lines.append(f"[{dotted}]")
        for key, value in strings:
            lines.append(f"{_toml_key(key)} = {json.dumps(value, ensure_ascii=False)}")
    for key, child in children:
        _append_toml_tables(lines, (*prefix, key), child)


def dump_prompt_config(data: Mapping[str, Any]) -> str:
    lines = [
        "# AMIE prompt configuration managed by the web prompt editor.",
        "# Use the per-agent reset action to restore values from prompts.example.toml.",
    ]
    _append_toml_tables(lines, (), data)
    return "\n".join(lines).rstrip() + "\n"


def save_prompt_config(path: Path, data: Mapping[str, Any]) -> None:
    example_path = default_prompt_config_path()
    if path.resolve() == example_path.resolve():
        raise PromptConfigError("不能覆盖内置默认提示词文件 prompts.example.toml")

    rendered = dump_prompt_config(data)
    try:
        parsed = tomllib.loads(rendered)
    except tomllib.TOMLDecodeError as exc:
        raise PromptConfigError(f"生成的提示词配置不是有效 TOML：{exc}") from exc
    if parsed != data:
        raise PromptConfigError("提示词配置序列化校验失败")

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except OSError as exc:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise PromptConfigError(f"无法保存提示词配置：{path}: {exc}") from exc
