from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .config import DEFAULT_CONFIG_PATH, ModelConfigError, Settings, load_model_catalog
from .llm import LLM, ModelAPIClient
from .simulation import SimulationSession


PACKAGE_DIR = Path(__file__).resolve().parent
STATIC_DIR = PACKAGE_DIR / "static"
PROJECT_DIR = PACKAGE_DIR.parents[1]
DISEASES_PATH = PROJECT_DIR / "malacards-diseases.json"


def load_diseases(path: Path = DISEASES_PATH) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        rows = json.load(handle)
    seen: set[str] = set()
    diseases: list[str] = []
    for row in rows:
        name = row.get("disease") if isinstance(row, dict) else None
        if isinstance(name, str) and name not in seen:
            seen.add(name)
            diseases.append(name)
    return diseases


def load_models(path: Path = DEFAULT_CONFIG_PATH) -> list[dict[str, Any]]:
    """Load browser-safe model metadata from the configured API catalog."""

    return load_model_catalog(path).public_models()


def create_app(
    *,
    llm: LLM | None = None,
    diseases: list[str] | None = None,
    models: list[dict[str, Any]] | None = None,
) -> FastAPI:
    settings = Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        catalog = None
        if models is None:
            catalog = load_model_catalog(settings.config_path, settings)
            visible_models = catalog.public_models()
            default_model = catalog.default_model
        else:
            visible_models = models
            if not visible_models:
                raise ModelConfigError("models 至少需要包含一个模型")
            model_names = {item["name"] for item in visible_models}
            default_model = (
                settings.model
                if settings.model and settings.model in model_names
                else visible_models[0]["name"]
            )

        if llm is None and catalog is None and not settings.endpoint:
            raise ModelConfigError("创建真实模型客户端时必须提供模型 API 配置")
        app.state.llm = llm or ModelAPIClient(settings, catalog=catalog)
        app.state.owns_llm = llm is None
        app.state.diseases = diseases if diseases is not None else load_diseases()
        app.state.models = visible_models
        app.state.model_names = {item["name"] for item in app.state.models}
        app.state.default_model = default_model
        yield
        if app.state.owns_llm:
            await app.state.llm.aclose()

    app = FastAPI(
        title="AMIE Self-play Prototype",
        version="0.5.0",
        lifespan=lifespan,
    )
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/diseases")
    async def disease_search(
        q: str = Query(default="", max_length=120), limit: int = Query(10, ge=1, le=20)
    ) -> dict[str, Any]:
        needle = q.strip().casefold()
        if not needle:
            return {"items": []}
        prefix: list[str] = []
        contains: list[str] = []
        for name in app.state.diseases:
            folded = name.casefold()
            if folded.startswith(needle):
                prefix.append(name)
            elif needle in folded:
                contains.append(name)
            if len(prefix) >= limit:
                break
        items = (prefix + contains)[:limit]
        return {"items": items}

    @app.get("/api/models")
    async def model_catalog() -> dict[str, Any]:
        return {
            "default_model": app.state.default_model,
            "items": app.state.models,
        }

    @app.websocket("/ws/simulation")
    async def simulation_socket(websocket: WebSocket) -> None:
        await websocket.accept()
        session = SimulationSession(
            app.state.llm,
            websocket.send_json,
            model_name=app.state.default_model,
        )
        try:
            while True:
                try:
                    command = await websocket.receive_json()
                except ValueError:
                    await websocket.send_json(
                        {"type": "error", "code": "invalid_command", "message": "消息必须是 JSON"}
                    )
                    continue
                action = command.get("action")
                if action == "start":
                    model_name = str(
                        command.get("model") or app.state.default_model
                    )
                    if model_name not in app.state.model_names:
                        await websocket.send_json(
                            {
                                "type": "error",
                                "code": "invalid_model",
                                "message": f"模型 {model_name} 不在模型 API 配置中",
                            }
                        )
                        continue
                    await session.launch_start(
                        str(command.get("condition", "")), model_name
                    )
                elif action == "refine":
                    await session.launch_refine()
                elif action == "stop":
                    await session.stop()
                else:
                    await websocket.send_json(
                        {
                            "type": "error",
                            "code": "invalid_command",
                            "message": "仅支持 start、refine 或 stop",
                        }
                    )
        except WebSocketDisconnect:
            await session.disconnect()
        finally:
            await session.disconnect()

    return app


app = create_app()
