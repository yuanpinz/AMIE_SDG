from __future__ import annotations

import asyncio
import os
from typing import Protocol, TypedDict

import httpx

from .config import (
    ModelAPI,
    ModelCatalog,
    ModelConfigError,
    Settings,
    expand_environment,
)


class ChatMessage(TypedDict):
    role: str
    content: str


class LLM(Protocol):
    async def complete(
        self,
        messages: list[ChatMessage],
        *,
        max_tokens: int = 1200,
        temperature: float = 0.2,
        model_name: str | None = None,
    ) -> str: ...


class ModelAPIError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        code: str = "llm_error",
        retryable: bool = False,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable
        self.status_code = status_code


def extract_text(data: object) -> str | None:
    if not isinstance(data, dict):
        return None
    result = data.get("result")
    if isinstance(result, str):
        return result
    choices = data.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        message = choices[0].get("message")
        if isinstance(message, dict) and isinstance(message.get("content"), str):
            return message["content"]
    return None


class ModelAPIClient:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        catalog: ModelCatalog | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.settings = settings or Settings.from_env()
        self.catalog = catalog
        self._owns_client = client is None
        timeout_seconds = (
            catalog.timeout_seconds if catalog is not None else self.settings.timeout_seconds
        )
        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(timeout_seconds)
        )

    def _configured_request(
        self, model_name: str | None
    ) -> tuple[str, str, str, dict[str, str], dict[str, object], int]:
        if self.catalog is None:
            endpoint = self.settings.endpoint
            api_model = model_name or self.settings.model
            if not endpoint or not api_model:
                raise ModelAPIError(
                    "未配置模型 API。请设置 config/model_apis.json。",
                    code="configuration_error",
                )
            return (
                endpoint,
                api_model,
                "model_name",
                {},
                {"enable_thinking": False},
                self.settings.max_retries,
            )

        selected_name = model_name or self.catalog.default_model
        try:
            model = self.catalog.by_name(selected_name)
            headers = self._request_headers(model)
        except ModelConfigError as exc:
            raise ModelAPIError(str(exc), code="configuration_error") from exc
        return (
            model.endpoint,
            model.api_model,
            model.model_field,
            headers,
            dict(model.body),
            self.catalog.max_retries,
        )

    @staticmethod
    def _request_headers(model: ModelAPI) -> dict[str, str]:
        headers = {
            key: expand_environment(value) for key, value in model.headers.items()
        }
        if model.api_key_env:
            api_key = os.getenv(model.api_key_env)
            if not api_key:
                raise ModelConfigError(
                    f"模型 {model.name!r} 需要环境变量 {model.api_key_env}"
                )
            headers[model.api_key_header] = f"{model.api_key_prefix}{api_key}"
        return headers

    async def complete(
        self,
        messages: list[ChatMessage],
        *,
        max_tokens: int = 1200,
        temperature: float = 0.2,
        model_name: str | None = None,
    ) -> str:
        endpoint, api_model, model_field, headers, extra_body, max_retries = (
            self._configured_request(model_name)
        )
        payload = {
            "messages": messages,
            "stream": False,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        payload[model_field] = api_model
        payload.update(extra_body)
        attempts = max_retries + 1
        for attempt in range(attempts):
            try:
                response = await self._client.post(
                    endpoint, json=payload, headers=headers
                )
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                if attempt + 1 < attempts:
                    await asyncio.sleep(0)
                    continue
                raise ModelAPIError(
                    f"模型请求失败：{exc}", code="network_error", retryable=True
                ) from exc

            if response.status_code >= 500:
                if attempt + 1 < attempts:
                    await asyncio.sleep(0)
                    continue
                raise ModelAPIError(
                    f"模型服务暂时不可用（HTTP {response.status_code}）",
                    code="server_error",
                    retryable=True,
                    status_code=response.status_code,
                )
            if response.status_code >= 400:
                detail = response.text[:500]
                raise ModelAPIError(
                    f"模型请求被拒绝（HTTP {response.status_code}）：{detail}",
                    code="api_error",
                    status_code=response.status_code,
                )

            try:
                data = response.json()
            except ValueError as exc:
                raise ModelAPIError(
                    "模型服务返回了无法解析的响应", code="invalid_response"
                ) from exc
            text = extract_text(data)
            if text is None or not text.strip():
                raise ModelAPIError(
                    "模型服务没有返回有效文本", code="empty_response"
                )
            return text.strip()
        raise AssertionError("unreachable")

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()
