from __future__ import annotations

import httpx
import pytest

from amie_self_play.config import ModelAPI, ModelCatalog, Settings
from amie_self_play.llm import ModelAPIClient, ModelAPIError


MESSAGES = [{"role": "user", "content": "hello"}]


@pytest.mark.asyncio
async def test_catalog_routes_model_and_adds_configured_auth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    monkeypatch.setenv("TEST_LLM_KEY", "secret-value")
    monkeypatch.setenv("TEST_TENANT", "tenant-a")
    catalog = ModelCatalog(
        default_model="public-name",
        models=(
            ModelAPI(
                name="public-name",
                real_name="Upstream name",
                endpoint="https://example.test/v1/chat/completions",
                api_model="upstream-model-id",
                api_key_env="TEST_LLM_KEY",
                headers={"X-Tenant": "${TEST_TENANT}"},
                body={"enable_thinking": False},
            ),
        ),
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = ModelAPIClient(Settings(), catalog=catalog, client=http_client)
        assert await client.complete(MESSAGES, model_name="public-name") == "ok"

    assert requests[0].url == "https://example.test/v1/chat/completions"
    assert requests[0].headers["Authorization"] == "Bearer secret-value"
    assert requests[0].headers["X-Tenant"] == "tenant-a"
    payload = __import__("json").loads(requests[0].content)
    assert payload["model"] == "upstream-model-id"
    assert payload["enable_thinking"] is False
    assert "model_name" not in payload


@pytest.mark.asyncio
async def test_catalog_reports_missing_api_key_before_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(200, json={"result": "unexpected"})

    monkeypatch.delenv("MISSING_LLM_KEY", raising=False)
    catalog = ModelCatalog(
        default_model="model-alpha",
        models=(
            ModelAPI(
                name="model-alpha",
                real_name="Model Alpha",
                endpoint="https://example.test/chat",
                api_model="upstream-alpha",
                api_key_env="MISSING_LLM_KEY",
            ),
        ),
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = ModelAPIClient(Settings(), catalog=catalog, client=http_client)
        with pytest.raises(ModelAPIError) as caught:
            await client.complete(MESSAGES)

    assert attempts == 0
    assert caught.value.code == "configuration_error"
    assert "MISSING_LLM_KEY" in str(caught.value)


@pytest.mark.asyncio
async def test_retries_once_on_5xx_and_sends_expected_gateway_payload() -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(503, text="busy")
        return httpx.Response(200, json={"result": "ok"})

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http_client:
        client = ModelAPIClient(
            Settings(endpoint="https://example.test/chat", model="model-alpha", max_retries=1),
            client=http_client,
        )
        result = await client.complete(MESSAGES, model_name="model-beta")

    assert result == "ok"
    assert len(requests) == 2
    payload = __import__("json").loads(requests[-1].content)
    assert payload["model_name"] == "model-beta"
    assert payload["enable_thinking"] is False
    assert payload["stream"] is False


@pytest.mark.asyncio
async def test_retries_once_on_timeout() -> None:
    attempts = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise httpx.ReadTimeout("slow", request=request)
        return httpx.Response(200, json={"choices": [{"message": {"content": "recovered"}}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = ModelAPIClient(
            Settings(
                endpoint="https://example.test/chat",
                model="model-alpha",
                max_retries=1,
            ),
            client=http_client,
        )
        assert await client.complete(MESSAGES) == "recovered"
    assert attempts == 2


@pytest.mark.asyncio
async def test_does_not_retry_4xx() -> None:
    attempts = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(400, text="bad request")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = ModelAPIClient(
            Settings(
                endpoint="https://example.test/chat",
                model="model-alpha",
                max_retries=1,
            ),
            client=http_client,
        )
        with pytest.raises(ModelAPIError) as caught:
            await client.complete(MESSAGES)
    assert attempts == 1
    assert caught.value.code == "api_error"
    assert caught.value.retryable is False
