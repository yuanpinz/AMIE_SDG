from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from amie_self_play.app import create_app, load_models

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
    assert "evaluation_completed" in event_types
    assert event_types[-1] == "round_review_completed"
    assert event_types.index("dialogue_completed") < event_types.index("ddx_completed")
    assert event_types.index("ddx_completed") < event_types.index("critique_completed")
    assert event_types.index("critique_completed") < event_types.index(
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
