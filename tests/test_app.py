from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

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


def test_health_page_and_local_disease_autocomplete() -> None:
    app = create_app(
        llm=FakeLLM(),
        diseases=["Carpal Tunnel Syndrome", "Carpal Bone Fracture", "Migraine"],
        models=TEST_MODELS,
    )
    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok"}
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


def test_prompt_admin_page_saves_and_resets_one_agent(
    tmp_path: Path, monkeypatch
) -> None:
    path = tmp_path / "prompts.toml"
    defaults = load_default_prompt_catalog()
    save_prompt_config(path, defaults.data)
    monkeypatch.setenv("AMIE_PROMPT_CONFIG", str(path))
    monkeypatch.delenv("AMIE_PROMPT_ADMIN_TOKEN", raising=False)
    app = create_app(llm=FakeLLM(), diseases=[], models=TEST_MODELS)

    with TestClient(app) as client:
        page = client.get("/admin/prompts")
        assert page.status_code == 200
        assert 'id="promptFields"' in page.text

        payload = client.get("/api/admin/prompts").json()
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
        assert "CUSTOM_ADMIN_DOCTOR_PROMPT" in app.state.prompts.data["agent"]["doctor"]["system"]
        assert "CUSTOM_ADMIN_DOCTOR_PROMPT" in load_prompt_catalog(path).data["agent"]["doctor"]["system"]

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


def test_prompt_admin_rejects_invalid_prompt_and_supports_token(
    tmp_path: Path, monkeypatch
) -> None:
    path = tmp_path / "prompts.toml"
    defaults = load_default_prompt_catalog()
    save_prompt_config(path, defaults.data)
    monkeypatch.setenv("AMIE_PROMPT_CONFIG", str(path))
    monkeypatch.setenv("AMIE_PROMPT_ADMIN_TOKEN", "test-admin-token")
    app = create_app(llm=FakeLLM(), diseases=[], models=TEST_MODELS)

    with TestClient(app) as client:
        assert client.get("/api/admin/prompts").status_code == 401
        headers = {"X-AMIE-ADMIN-TOKEN": "test-admin-token"}
        payload = client.get("/api/admin/prompts", headers=headers).json()
        doctor = next(item for item in payload["agents"] if item["name"] == "doctor")
        values = {field["name"]: field["value"] for field in doctor["fields"]}
        values["system"] = "[ROLE:DOCTOR]\n缺少必要变量"

        response = client.put(
            "/api/admin/prompts/doctor",
            headers=headers,
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


def test_websocket_rejects_model_outside_guide_catalog() -> None:
    app = create_app(llm=FakeLLM(), diseases=[], models=TEST_MODELS)
    with TestClient(app) as client:
        with client.websocket_connect("/ws/simulation") as websocket:
            websocket.send_json(
                {"action": "start", "condition": "腕管综合征", "model": "unknown"}
            )
            event = websocket.receive_json()
    assert event["type"] == "error"
    assert event["code"] == "invalid_model"
