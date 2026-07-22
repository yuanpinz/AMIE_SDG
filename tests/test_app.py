from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from amie_self_play.app import create_app, load_models
from amie_self_play.prompt_config import (
    load_default_prompt_catalog,
    load_prompt_catalog,
    save_prompt_config,
)

from conftest import FakeLLM
from conftest import (
    ddx_payload,
    evaluation_responses,
    json_text,
    moderator_payload,
    vignette_payload,
)


TEST_MODELS = [
    {
        "name": "model-alpha",
        "real_name": "example-alpha",
        "context_length": 1048576,
        "support_stream": True,
        "image_input": False,
        "audio_output": False,
    },
    {
        "name": "model-beta",
        "real_name": "example-beta",
        "context_length": 128000,
        "support_stream": True,
        "image_input": False,
        "audio_output": False,
    },
]


def register_user(
    client: TestClient,
    username: str = "test-user",
    password: str = "test-password-123",
) -> None:
    response = client.post(
        "/api/auth/register", json={"username": username, "password": password}
    )
    assert response.status_code == 201
    assert response.json()["user"]["username"] == username


def test_health_page_and_local_disease_autocomplete() -> None:
    app = create_app(
        llm=FakeLLM(),
        diseases=["Carpal Tunnel Syndrome", "Carpal Bone Fracture", "Migraine"],
        models=TEST_MODELS,
    )
    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok"}
        redirect = client.get("/", follow_redirects=False)
        assert redirect.status_code == 303
        assert redirect.headers["location"] == "/login?next=/"
        login_page = client.get("/login")
        assert login_page.status_code == 200
        assert 'id="authForm"' in login_page.text
        register_user(client)
        page = client.get("/")
        assert page.status_code == 200
        assert 'href="/admin/prompts"' in page.text
        assert "把一次问诊" in page.text
        assert "Model-based proxy" in page.text
        assert "不能替代真实患者或专科医生评价" in page.text
        assert "Accepted Differential" not in page.text
        assert page.text.index('id="evaluationPanel"') < page.text.index(
            'id="roundActions"'
        )
        result = client.get("/api/diseases", params={"q": "carpal", "limit": 10})
        assert result.json()["items"] == [
            "Carpal Tunnel Syndrome",
            "Carpal Bone Fracture",
        ]
        catalog = client.get("/api/models").json()
        assert catalog["default_model"] == "model-alpha"
        assert [item["name"] for item in catalog["items"]] == [
            "model-alpha",
            "model-beta",
        ]


def test_prompt_admin_page_saves_and_resets_one_agent() -> None:
    defaults = load_default_prompt_catalog()
    app = create_app(llm=FakeLLM(), diseases=[], models=TEST_MODELS)

    with TestClient(app) as client:
        register_user(client)
        page = client.get("/admin/prompts")
        assert page.status_code == 200
        assert 'id="promptFields"' in page.text

        payload = client.get("/api/admin/prompts").json()
        path = Path(payload["source"])
        assert payload["owner"] == "test-user"
        assert path.stat().st_mode & 0o777 == 0o600
        assert path.parent.stat().st_mode & 0o777 == 0o700
        assert len(payload["agents"]) == 11
        doctor = next(item for item in payload["agents"] if item["name"] == "doctor")
        values = {field["name"]: field["value"] for field in doctor["fields"]}
        values["system"] += "\nCUSTOM_ADMIN_DOCTOR_PROMPT"

        saved = client.put(
            "/api/admin/prompts/doctor", json={"values": values}
        )
        assert saved.status_code == 200
        saved_doctor = next(
            item for item in saved.json()["agents"] if item["name"] == "doctor"
        )
        assert saved_doctor["modified"] is True
        assert (
            "CUSTOM_ADMIN_DOCTOR_PROMPT"
            in load_prompt_catalog(path).data["agent"]["doctor"]["system"]
        )

        reset = client.post("/api/admin/prompts/doctor/reset")
        assert reset.status_code == 200
        reset_doctor = next(
            item for item in reset.json()["agents"] if item["name"] == "doctor"
        )
        assert reset_doctor["modified"] is False
        assert (
            load_prompt_catalog(path).data["agent"]["doctor"]
            == defaults.data["agent"]["doctor"]
        )


def test_accounts_validate_login_logout_and_duplicate_user() -> None:
    app = create_app(llm=FakeLLM(), diseases=[], models=TEST_MODELS)

    with TestClient(app) as client:
        rejected = client.post(
            "/api/auth/register",
            json={"username": "ab", "password": "short"},
        )
        assert rejected.status_code == 422

        register = client.post(
            "/api/auth/register",
            json={"username": "Alice", "password": "alice-password-123"},
        )
        assert register.status_code == 201
        assert "HttpOnly" in register.headers["set-cookie"]
        assert "alice-password-123" not in register.headers["set-cookie"]
        assert client.get("/api/auth/me").json()["user"]["username"] == "Alice"

        duplicate = client.post(
            "/api/auth/register",
            json={"username": "alice", "password": "another-password-123"},
        )
        assert duplicate.status_code == 409

        assert client.post("/api/auth/logout").status_code == 200
        assert client.get("/api/auth/me").status_code == 401
        wrong = client.post(
            "/api/auth/login",
            json={"username": "Alice", "password": "wrong-password"},
        )
        assert wrong.status_code == 401
        login = client.post(
            "/api/auth/login",
            json={"username": "alice", "password": "alice-password-123"},
        )
        assert login.status_code == 200
        assert client.get("/").status_code == 200


def test_auth_attempts_are_rate_limited() -> None:
    app = create_app(llm=FakeLLM(), diseases=[], models=TEST_MODELS)

    with TestClient(app) as client:
        for _ in range(10):
            response = client.post(
                "/api/auth/login",
                json={"username": "missing-user", "password": "wrong-password"},
            )
            assert response.status_code == 401
        limited = client.post(
            "/api/auth/login",
            json={"username": "missing-user", "password": "wrong-password"},
        )
        assert limited.status_code == 429
        assert limited.headers["retry-after"] == "60"


def test_users_have_isolated_prompt_files() -> None:
    app = create_app(llm=FakeLLM(), diseases=[], models=TEST_MODELS)

    with TestClient(app) as client:
        register_user(client, "alice", "alice-password-123")
        alice_payload = client.get("/api/admin/prompts").json()
        alice_path = Path(alice_payload["source"])
        alice_doctor = next(
            item for item in alice_payload["agents"] if item["name"] == "doctor"
        )
        alice_values = {
            field["name"]: field["value"] for field in alice_doctor["fields"]
        }
        alice_values["system"] += "\nALICE_PRIVATE_PROMPT"
        assert client.put(
            "/api/admin/prompts/doctor", json={"values": alice_values}
        ).status_code == 200
        client.post("/api/auth/logout")

        register_user(client, "bob", "bob-password-12345")
        bob_payload = client.get("/api/admin/prompts").json()
        bob_path = Path(bob_payload["source"])
        bob_doctor = next(
            item for item in bob_payload["agents"] if item["name"] == "doctor"
        )
        assert all(
            "ALICE_PRIVATE_PROMPT" not in field["value"]
            for field in bob_doctor["fields"]
        )
        assert bob_path != alice_path
        client.post("/api/auth/logout")

        assert client.post(
            "/api/auth/login",
            json={"username": "alice", "password": "alice-password-123"},
        ).status_code == 200
        restored = client.get("/api/admin/prompts").json()
        restored_doctor = next(
            item for item in restored["agents"] if item["name"] == "doctor"
        )
        assert any(
            "ALICE_PRIVATE_PROMPT" in field["value"]
            for field in restored_doctor["fields"]
        )
        assert alice_path.is_file()
        assert bob_path.is_file()


def test_new_user_uses_configured_prompt_seed(tmp_path: Path, monkeypatch) -> None:
    defaults = load_default_prompt_catalog()
    seed_data = copy.deepcopy(defaults.data)
    seed_data["agent"]["doctor"]["system"] += "\nORGANIZATION_PROMPT_SEED"
    seed_path = tmp_path / "organization-prompts.toml"
    save_prompt_config(seed_path, seed_data)
    monkeypatch.setenv("AMIE_PROMPT_CONFIG", str(seed_path))
    app = create_app(llm=FakeLLM(), diseases=[], models=TEST_MODELS)

    with TestClient(app) as client:
        register_user(client)
        payload = client.get("/api/admin/prompts").json()
        doctor = next(item for item in payload["agents"] if item["name"] == "doctor")
        assert any(
            "ORGANIZATION_PROMPT_SEED" in field["value"]
            for field in doctor["fields"]
        )
        reset = client.post("/api/admin/prompts/doctor/reset")
        reset_doctor = next(
            item for item in reset.json()["agents"] if item["name"] == "doctor"
        )
        assert all(
            "ORGANIZATION_PROMPT_SEED" not in field["value"]
            for field in reset_doctor["fields"]
        )


def test_prompt_admin_rejects_invalid_prompt() -> None:
    defaults = load_default_prompt_catalog()
    app = create_app(llm=FakeLLM(), diseases=[], models=TEST_MODELS)

    with TestClient(app) as client:
        register_user(client)
        payload = client.get("/api/admin/prompts").json()
        path = Path(payload["source"])
        doctor = next(item for item in payload["agents"] if item["name"] == "doctor")
        values = {field["name"]: field["value"] for field in doctor["fields"]}
        values["system"] = "[ROLE:DOCTOR]\n缺少必要变量"

        response = client.put(
            "/api/admin/prompts/doctor",
            json={"values": values},
        )

        assert response.status_code == 422
        assert "improvement_context" in response.json()["detail"]
        assert load_prompt_catalog(path).data == defaults.data


def test_model_api_config_is_loaded_without_exposing_connection_details(
    tmp_path: Path,
) -> None:
    path = tmp_path / "models.json"
    path.write_text(
        json.dumps(
            {
                "default_model": "public-id",
                "models": [
                    {
                        "name": "public-id",
                        "real_name": "API model name",
                        "endpoint": "https://example.test/v1/chat/completions",
                        "model": "private-upstream-id",
                        "api_key_env": "SECRET_API_KEY",
                        "context_length": 128000,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    models = load_models(path)
    assert models == [
        {
            "name": "public-id",
            "real_name": "API model name",
            "context_length": 128000,
            "support_stream": False,
            "image_input": False,
        }
    ]
    assert "endpoint" not in models[0]
    assert "api_key_env" not in models[0]


def test_websocket_start_runs_complete_event_protocol() -> None:
    fake = FakeLLM(
        VIGNETTE=[json_text(vignette_payload("腕管综合征"))],
        PATIENT=["右手夜间麻木。"],
        DOCTOR=["考虑腕管综合征，建议线下面诊并夜间佩戴腕托。"],
        MODERATOR=[json_text(moderator_payload())],
        DDX=[json_text(ddx_payload())],
        CRITIC=["问诊完整，可更早询问工作史。"],
        **evaluation_responses(),
    )
    app = create_app(llm=fake, diseases=[], models=TEST_MODELS)
    with TestClient(app) as client:
        register_user(client)
        payload = client.get("/api/admin/prompts").json()
        doctor = next(item for item in payload["agents"] if item["name"] == "doctor")
        values = {field["name"]: field["value"] for field in doctor["fields"]}
        values["system"] += "\nUSER_WEBSOCKET_PROMPT"
        assert client.put(
            "/api/admin/prompts/doctor", json={"values": values}
        ).status_code == 200
        with client.websocket_connect("/ws/simulation") as websocket:
            websocket.send_json(
                {
                    "action": "start",
                    "condition": "腕管综合征",
                    "model": "model-beta",
                }
            )
            events = []
            while True:
                event = websocket.receive_json()
                events.append(event)
                if event["type"] == "round_review_completed":
                    break

    event_types = [event["type"] for event in events]
    assert event_types[0] == "simulation_started"
    assert "vignette_completed" in event_types
    assert "moderator_result" in event_types
    assert "dialogue_completed" in event_types
    assert "ddx_completed" in event_types
    assert "critique_completed" in event_types
    assert "round_review_ready" in event_types
    assert "evaluation_completed" in event_types
    assert event_types[-1] == "round_review_completed"
    assert event_types.index("dialogue_completed") < event_types.index("ddx_completed")
    assert event_types.index("ddx_completed") < event_types.index("critique_completed")
    assert event_types.index("critique_completed") < event_types.index(
        "round_review_ready"
    )
    assert event_types.index("round_review_ready") < event_types.index(
        "evaluation_completed"
    )
    evaluation = next(
        event for event in events if event["type"] == "evaluation_completed"
    )
    assert evaluation["status"] == "complete"
    assert set(evaluation) >= {
        "accuracy",
        "patient_actor",
        "specialist",
        "auto_paces",
        "errors",
    }
    messages = [event for event in events if event["type"] == "message_completed"]
    assert [event["role"] for event in messages] == ["doctor", "patient", "doctor"]
    assert all("content_html" in event for event in messages if event["role"] == "doctor")
    assert all("content_html" not in event for event in messages if event["role"] == "patient")
    assert all(call["model_name"] == "model-beta" for call in fake.calls)
    doctor_calls = [call for call in fake.calls if call["role"] == "DOCTOR"]
    assert "USER_WEBSOCKET_PROMPT" in doctor_calls[0]["messages"][0]["content"]


def test_websocket_requires_authenticated_user() -> None:
    app = create_app(llm=FakeLLM(), diseases=[], models=TEST_MODELS)
    with TestClient(app) as client:
        with pytest.raises(WebSocketDisconnect) as captured:
            with client.websocket_connect("/ws/simulation"):
                pass
    assert captured.value.code == 4401


def test_websocket_rejects_model_outside_guide_catalog() -> None:
    app = create_app(llm=FakeLLM(), diseases=[], models=TEST_MODELS)
    with TestClient(app) as client:
        register_user(client)
        with client.websocket_connect("/ws/simulation") as websocket:
            websocket.send_json(
                {"action": "start", "condition": "腕管综合征", "model": "unknown"}
            )
            event = websocket.receive_json()
    assert event["type"] == "error"
    assert event["code"] == "invalid_model"
