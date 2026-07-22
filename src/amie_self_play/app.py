from __future__ import annotations

import asyncio
import json
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import (
    FastAPI,
    HTTPException,
    Query,
    Request,
    Response,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import DEFAULT_CONFIG_PATH, ModelConfigError, Settings, load_model_catalog
from .llm import LLM, ModelAPIClient
from .prompt_config import (
    PromptConfigError,
    agent_prompt_payload,
    load_default_prompt_catalog,
    load_prompt_catalog,
    reset_agent_prompt_data,
    save_prompt_config,
    update_agent_prompt_data,
)
from .simulation import SimulationSession
from .user_store import (
    InvalidCredentialsError,
    User,
    UserStore,
    UserStoreError,
    UsernameTakenError,
)


PACKAGE_DIR = Path(__file__).resolve().parent
STATIC_DIR = PACKAGE_DIR / "static"
PROJECT_DIR = PACKAGE_DIR.parents[1]
DISEASES_PATH = PROJECT_DIR / "malacards-diseases.json"
SESSION_COOKIE = "amie_session"
SESSION_MAX_AGE = 7 * 24 * 60 * 60
AUTH_ATTEMPT_LIMIT = 10
AUTH_ATTEMPT_WINDOW_SECONDS = 60
AUTH_CLIENT_BUCKET_LIMIT = 4096


class PromptAgentUpdate(BaseModel):
    values: dict[str, str]


class AuthCredentials(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=128)


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
    user_store = UserStore(settings.user_data_dir)
    auth_attempts: dict[str, deque[float]] = defaultdict(deque)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        user_store.initialize()
        prompt_defaults = load_default_prompt_catalog()
        prompt_seed = (
            load_prompt_catalog(settings.prompt_config_path)
            if settings.prompt_config_path.is_file()
            else prompt_defaults
        )
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
        app.state.prompt_defaults = prompt_defaults
        app.state.prompt_seed = prompt_seed
        app.state.prompt_config_lock = asyncio.Lock()
        app.state.user_store = user_store
        app.state.model_names = {item["name"] for item in app.state.models}
        app.state.default_model = default_model
        yield
        if app.state.owns_llm:
            await app.state.llm.aclose()

    app = FastAPI(
        title="AMIE Self-play Prototype",
        version="0.6.0",
        lifespan=lifespan,
    )
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    def request_user(request: Request) -> User | None:
        return user_store.user_for_session(request.cookies.get(SESSION_COOKIE))

    def require_user(request: Request) -> User:
        user = request_user(request)
        if user is None:
            raise HTTPException(status_code=401, detail="请先登录")
        return user

    def enforce_auth_rate_limit(request: Request) -> None:
        client_host = request.client.host if request.client else "unknown"
        now = time.monotonic()
        cutoff = now - AUTH_ATTEMPT_WINDOW_SECONDS
        if (
            client_host not in auth_attempts
            and len(auth_attempts) >= AUTH_CLIENT_BUCKET_LIMIT
        ):
            stale_hosts = [
                host
                for host, bucket in auth_attempts.items()
                if not bucket or bucket[-1] <= cutoff
            ]
            for host in stale_hosts:
                del auth_attempts[host]
            if len(auth_attempts) >= AUTH_CLIENT_BUCKET_LIMIT:
                oldest_host = min(
                    auth_attempts,
                    key=lambda host: auth_attempts[host][-1],
                )
                del auth_attempts[oldest_host]
        attempts = auth_attempts[client_host]
        while attempts and attempts[0] <= cutoff:
            attempts.popleft()
        if len(attempts) >= AUTH_ATTEMPT_LIMIT:
            raise HTTPException(
                status_code=429,
                detail="操作过于频繁，请稍后再试",
                headers={"Retry-After": str(AUTH_ATTEMPT_WINDOW_SECONDS)},
            )
        attempts.append(now)

    def set_session_cookie(request: Request, response: Response, token: str) -> None:
        response.set_cookie(
            key=SESSION_COOKIE,
            value=token,
            max_age=SESSION_MAX_AGE,
            httponly=True,
            samesite="strict",
            secure=request.url.scheme == "https",
            path="/",
        )

    def clear_session_cookie(request: Request, response: Response) -> None:
        response.delete_cookie(
            key=SESSION_COOKIE,
            httponly=True,
            samesite="strict",
            secure=request.url.scheme == "https",
            path="/",
        )

    def user_prompt_path(user: User) -> Path:
        return user_store.prompt_path(user)

    def save_user_prompt_config(user: User, data: dict[str, Any]) -> Path:
        path = user_prompt_path(user)
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        path.parent.chmod(0o700)
        save_prompt_config(path, data)
        path.chmod(0o600)
        return path

    def user_prompt_catalog(user: User):
        path = user_prompt_path(user)
        if not path.is_file():
            save_user_prompt_config(user, app.state.prompt_seed.data)
        return load_prompt_catalog(path)

    def safe_next_path(value: str) -> str:
        return value if value.startswith("/") and not value.startswith("//") else "/"

    @app.get("/login", include_in_schema=False)
    async def login_page(
        request: Request, next_path: str = Query(default="/", alias="next")
    ) -> Response:
        if request_user(request) is not None:
            return RedirectResponse(url=safe_next_path(next_path), status_code=303)
        return FileResponse(STATIC_DIR / "auth.html")

    @app.get("/", include_in_schema=False)
    async def index(request: Request) -> Response:
        if request_user(request) is None:
            return RedirectResponse(url="/login?next=/", status_code=303)
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/admin/prompts", include_in_schema=False)
    async def prompt_admin_page(request: Request) -> Response:
        if request_user(request) is None:
            return RedirectResponse(
                url="/login?next=/admin/prompts", status_code=303
            )
        return FileResponse(STATIC_DIR / "prompt-config.html")

    @app.post("/api/auth/register", status_code=201)
    async def register(
        credentials: AuthCredentials, request: Request, response: Response
    ) -> dict[str, Any]:
        enforce_auth_rate_limit(request)
        user: User | None = None
        try:
            user = user_store.register(credentials.username, credentials.password)
            save_user_prompt_config(user, app.state.prompt_seed.data)
        except UsernameTakenError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except UserStoreError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except PromptConfigError:
            if user is not None:
                user_store.delete_user(user)
            raise
        assert user is not None
        token = user_store.create_session(user)
        set_session_cookie(request, response, token)
        return {"authenticated": True, "user": {"username": user.username}}

    @app.post("/api/auth/login")
    async def login(
        credentials: AuthCredentials, request: Request, response: Response
    ) -> dict[str, Any]:
        enforce_auth_rate_limit(request)
        try:
            user = user_store.authenticate(credentials.username, credentials.password)
        except (InvalidCredentialsError, UserStoreError) as exc:
            raise HTTPException(status_code=401, detail="用户名或密码错误") from exc
        token = user_store.create_session(user)
        set_session_cookie(request, response, token)
        return {"authenticated": True, "user": {"username": user.username}}

    @app.post("/api/auth/logout")
    async def logout(request: Request, response: Response) -> dict[str, bool]:
        user_store.delete_session(request.cookies.get(SESSION_COOKIE))
        clear_session_cookie(request, response)
        return {"authenticated": False}

    @app.get("/api/auth/me")
    async def current_account(request: Request) -> dict[str, Any]:
        user = require_user(request)
        return {"authenticated": True, "user": {"username": user.username}}

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

    def prompt_admin_response(user: User) -> dict[str, Any]:
        catalog = user_prompt_catalog(user)
        return {
            "source": str(user_prompt_path(user).resolve()),
            "owner": user.username,
            "agents": agent_prompt_payload(
                catalog, app.state.prompt_defaults
            ),
        }

    @app.get("/api/admin/prompts")
    async def get_prompt_admin(request: Request) -> dict[str, Any]:
        user = require_user(request)
        return prompt_admin_response(user)

    @app.put("/api/admin/prompts/{agent}")
    async def save_agent_prompts(
        agent: str, update: PromptAgentUpdate, request: Request
    ) -> dict[str, Any]:
        user = require_user(request)
        async with app.state.prompt_config_lock:
            try:
                catalog = user_prompt_catalog(user)
                data = update_agent_prompt_data(
                    catalog,
                    app.state.prompt_defaults,
                    agent,
                    update.values,
                )
                save_user_prompt_config(user, data)
            except PromptConfigError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
        response = prompt_admin_response(user)
        response["message"] = f"{agent} 提示词已保存，你的新会话将使用新配置"
        return response

    @app.post("/api/admin/prompts/{agent}/reset")
    async def reset_agent_prompts(agent: str, request: Request) -> dict[str, Any]:
        user = require_user(request)
        async with app.state.prompt_config_lock:
            try:
                catalog = user_prompt_catalog(user)
                data = reset_agent_prompt_data(
                    catalog, app.state.prompt_defaults, agent
                )
                save_user_prompt_config(user, data)
            except PromptConfigError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
        response = prompt_admin_response(user)
        response["message"] = f"{agent} 已恢复为内置默认提示词"
        return response

    @app.websocket("/ws/simulation")
    async def simulation_socket(websocket: WebSocket) -> None:
        session_token = websocket.cookies.get(SESSION_COOKIE)
        user = user_store.user_for_session(session_token)
        if user is None:
            await websocket.close(code=4401, reason="请先登录")
            return
        await websocket.accept()
        session = SimulationSession(
            app.state.llm,
            websocket.send_json,
            model_name=app.state.default_model,
            prompts=user_prompt_catalog(user),
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
                current_user = user_store.user_for_session(session_token)
                if current_user is None:
                    await websocket.send_json(
                        {"type": "error", "code": "auth_required", "message": "登录已失效"}
                    )
                    await websocket.close(code=4401, reason="登录已失效")
                    break
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
                    if not session.busy:
                        try:
                            session.prompts = user_prompt_catalog(current_user)
                        except PromptConfigError as exc:
                            await websocket.send_json(
                                {
                                    "type": "error",
                                    "code": "prompt_config_error",
                                    "message": str(exc),
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
