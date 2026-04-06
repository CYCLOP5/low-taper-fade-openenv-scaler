from __future__ import annotations

from contextlib import asynccontextmanager
import json
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from uuid import uuid4

from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import WebSocket
from fastapi import WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from sysadmin_env.models import Action
from sysadmin_env.models import EnvironmentState
from sysadmin_env.models import Observation
from sysadmin_env.models import ResetRequest
from sysadmin_env.models import StepRequest
from sysadmin_env.models import StepResult
from sysadmin_env.models import TaskScenarioDefinition
from sysadmin_env.rewards import EpisodeRewardState
from sysadmin_env.rewards import RewardEngine
from sysadmin_env.rewards import build_reward_engine
from sysadmin_env.sandbox import CommandResult
from sysadmin_env.sandbox import Sandbox
from sysadmin_env.tasks import TASK_MODULES
from sysadmin_env.tasks import build_task_registry


@dataclass
class EpisodeState:
    task_id: str
    sandbox: Sandbox
    reward_state: EpisodeRewardState
    max_steps: int
    step_number: int = 0


@dataclass
class HttpSessionState:
    episode_id: str | None = None
    episode: EpisodeState | None = None
    last_observation: Observation | None = None
    last_state: EnvironmentState | None = None


class EpisodeManager:
    def __init__(self, base_dir: str | Path | None = None) -> None:
        self._task_root = TemporaryDirectory(prefix="sysadmin_env_tasks_")
        self._task_registry = build_task_registry(self._task_root.name)
        self._reward_engine = build_reward_engine(self._task_registry)
        self._task_ids = list(self._task_registry)
        self._next_task_index = 0
        self._overlay_root = Path(base_dir).resolve() if base_dir is not None else None
        self._overlay_counter = 0
        if self._overlay_root is not None:
            (self._overlay_root / "runtime").mkdir(parents=True, exist_ok=True)
        self._prepare_task_filesystems()

    @property
    def task_registry(self) -> dict[str, TaskScenarioDefinition]:
        return self._task_registry

    @property
    def reward_engine(self) -> RewardEngine:
        return self._reward_engine

    def available_tasks(self) -> list[dict[str, Any]]:
        return [
            {
                "task_id": definition.metadata.task_id,
                "difficulty": definition.metadata.difficulty.value,
                "description": definition.metadata.description,
                "max_steps": definition.metadata.max_steps,
                "time_limit": definition.metadata.time_limit,
            }
            for definition in self._task_registry.values()
        ]

    def start_episode(self, task_id: str | None = None) -> EpisodeState:
        selected_task_id = task_id or self._select_next_task_id()
        print(f"episode start requested {selected_task_id}")
        if selected_task_id not in self._task_registry:
            raise KeyError(selected_task_id)

        definition = self._task_registry[selected_task_id]
        task_module = TASK_MODULES[selected_task_id]
        task_root = Path(definition.metadata.base_filesystem_path)
        print(f"episode fault inject start {selected_task_id}")
        task_module.inject_fault(task_root)

        sandbox = Sandbox(
            task_root,
            timeout=definition.metadata.time_limit,
            isolate_network=definition.requires_network_isolation,
            overlay_base_dir=self._allocate_overlay_dir(selected_task_id),
        )
        print(f"episode sandbox create start {selected_task_id}")
        sandbox.create()
        print(f"episode sandbox create complete {selected_task_id}")

        runtime_root = _runtime_root_for_definition(sandbox, definition)
        print(f"episode runtime root ready {runtime_root}")
        _synchronize_task_runtime(task_module, runtime_root)
        reward_state = self._reward_engine.start_episode(selected_task_id, runtime_root=runtime_root)
        print(f"episode reward state ready {selected_task_id}")

        return EpisodeState(
            task_id=selected_task_id,
            sandbox=sandbox,
            reward_state=reward_state,
            max_steps=definition.metadata.max_steps,
        )

    def cleanup_episode(self, episode: EpisodeState | None) -> None:
        if episode is None:
            return
        episode.sandbox.destroy()

    def shutdown(self) -> None:
        self._task_root.cleanup()

    def _prepare_task_filesystems(self) -> None:
        for task_id, module in TASK_MODULES.items():
            task_root = Path(self._task_registry[task_id].metadata.base_filesystem_path)
            task_root.mkdir(parents=True, exist_ok=True)
            module.prepare_filesystem(task_root)

    def _select_next_task_id(self) -> str:
        task_id = self._task_ids[self._next_task_index % len(self._task_ids)]
        self._next_task_index += 1
        return task_id

    def _allocate_overlay_dir(self, task_id: str) -> str | None:
        if self._overlay_root is None:
            return None
        overlay_dir = self._overlay_root / "runtime" / f"{task_id}_{self._overlay_counter}"
        self._overlay_counter += 1
        overlay_dir.mkdir(parents=True, exist_ok=True)
        return str(overlay_dir)


def create_app() -> FastAPI:
    manager = EpisodeManager(base_dir=Path.cwd() / "assets")
    web_metadata_payload = _build_web_metadata()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.episode_manager = manager
        try:
            yield
        finally:
            session: HttpSessionState = app.state.http_session
            if session.episode is not None:
                manager.cleanup_episode(session.episode)
            manager.shutdown()

    app = FastAPI(lifespan=lifespan)
    app.state.episode_manager = manager
    app.state.http_session = HttpSessionState()

    async def reset_episode(payload: ResetRequest | None = None) -> StepResult:
        manager: EpisodeManager = app.state.episode_manager
        session: HttpSessionState = app.state.http_session
        if session.episode is not None:
            print("reset cleanup previous episode")
            manager.cleanup_episode(session.episode)

        requested_task_id = payload.task_id if payload is not None else None
        print(f"reset requested task {requested_task_id or 'auto'}")
        try:
            episode = manager.start_episode(task_id=requested_task_id)
        except KeyError as exc:
            print("reset failed unknown task")
            raise HTTPException(status_code=404, detail="unknown task id") from exc
        except Exception as exc:
            print(f"reset failed {type(exc).__name__.lower()}")
            raise

        observation = Observation(
            stdout="",
            stderr="",
            exit_code=0,
            working_directory=str(getattr(episode.sandbox, "merged_root", Path("/"))),
            execution_time=0.0,
            reward=0.0,
            done=False,
            step_number=0,
            max_steps=episode.max_steps,
        )
        state = _build_environment_state(episode, uuid4().hex, observation)
        session.episode_id = state.episode_id
        session.episode = episode
        session.last_observation = observation
        session.last_state = state
        print(f"reset complete {state.task_id}")
        return StepResult(observation=observation, state=state)

    async def step_episode(payload: StepRequest) -> StepResult:
        manager: EpisodeManager = app.state.episode_manager
        session: HttpSessionState = app.state.http_session
        if session.episode is None or session.episode_id is None:
            raise HTTPException(status_code=409, detail="episode not initialized")

        command_result = await session.episode.sandbox.execute_async(payload.action.command)
        observation = _build_observation(manager, session.episode, payload.action.command, command_result)
        state = _build_environment_state(session.episode, session.episode_id, observation)
        session.last_observation = observation
        session.last_state = state
        if observation.done:
            manager.cleanup_episode(session.episode)
            session.episode = None
        return StepResult(observation=observation, state=state)

    @app.get("/health")
    async def health() -> JSONResponse:
        return JSONResponse({"status": "ok"})

    @app.post("/reset", response_model=StepResult)
    async def reset(payload: ResetRequest | None = None) -> StepResult:
        return await reset_episode(payload)

    @app.post("/step", response_model=StepResult)
    async def step(payload: StepRequest) -> StepResult:
        return await step_episode(payload)

    @app.get("/state", response_model=EnvironmentState)
    async def state() -> EnvironmentState:
        session: HttpSessionState = app.state.http_session
        if session.last_state is None:
            raise HTTPException(status_code=404, detail="episode not initialized")
        return session.last_state

    @app.get("/web", response_class=HTMLResponse)
    @app.get("/web/", response_class=HTMLResponse)
    async def web_interface() -> str:
        return _render_web_interface_html()

    @app.get("/web/metadata")
    async def web_metadata() -> JSONResponse:
        return JSONResponse(web_metadata_payload)

    @app.post("/web/reset")
    async def web_reset(payload: ResetRequest | None = None) -> JSONResponse:
        result = await reset_episode(payload)
        return JSONResponse(_build_web_step_result(result))

    @app.post("/web/step")
    async def web_step(payload: dict[str, Any]) -> JSONResponse:
        result = await step_episode(_parse_web_step_request(payload))
        return JSONResponse(_build_web_step_result(result))

    @app.get("/web/state")
    async def web_state() -> JSONResponse:
        session: HttpSessionState = app.state.http_session
        return JSONResponse(_build_web_state(session))

    @app.get("/tasks")
    async def tasks() -> JSONResponse:
        manager: EpisodeManager = app.state.episode_manager
        return JSONResponse({"tasks": manager.available_tasks()})

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket) -> None:
        await websocket.accept()
        manager: EpisodeManager = app.state.episode_manager
        episode: EpisodeState | None = None

        try:
            requested_task_id = websocket.query_params.get("task_id")
            try:
                episode = manager.start_episode(task_id=requested_task_id)
            except KeyError:
                await _send_error(websocket, "invalid_task", "unknown task id")
                await websocket.close(code=1008)
                return
            await _send_episode_started(websocket, manager, episode)

            while True:
                raw_message = await websocket.receive_text()
                action = _parse_action(raw_message)
                if action is None:
                    await _send_error(websocket, "invalid_action", "malformed action json")
                    continue

                if not action.command.strip():
                    await _send_error(websocket, "invalid_action", "command must not be empty")
                    continue

                command_result = await episode.sandbox.execute_async(action.command)
                observation = _build_observation(manager, episode, action.command, command_result)
                await websocket.send_json({
                    "type": "observation",
                    "task_id": episode.task_id,
                    "observation": observation.model_dump(),
                })

                if observation.done:
                    print(f"episode complete {episode.task_id} reward {observation.reward:.3f}")
                    manager.cleanup_episode(episode)
                    episode = None
                    break
        except WebSocketDisconnect:
            if episode is not None:
                manager.cleanup_episode(episode)
        except Exception:
            if episode is not None:
                manager.cleanup_episode(episode)
            raise

    return app


def _parse_action(raw_message: str) -> Action | None:
    try:
        payload = json.loads(raw_message)
    except json.JSONDecodeError:
        return None

    try:
        return Action.model_validate(payload)
    except ValidationError:
        return None


def _build_observation(
    manager: EpisodeManager,
    episode: EpisodeState,
    command: str,
    command_result: CommandResult,
) -> Observation:
    definition = manager.task_registry[episode.task_id]
    task_module = TASK_MODULES[episode.task_id]
    runtime_root = _runtime_root_for_definition(episode.sandbox, definition)
    _apply_task_runtime_updates(task_module, runtime_root, command, command_result)

    computation = manager.reward_engine.evaluate_action(episode.reward_state, command)
    episode.step_number += 1
    done = computation.task_state.done or computation.catastrophic or episode.step_number >= episode.max_steps
    if done:
        episode.reward_state.done = True

    stderr = command_result.stderr
    if command_result.timed_out:
        stderr = _merge_stderr(stderr, "command execution timed out")

    return Observation(
        stdout=command_result.stdout,
        stderr=stderr,
        exit_code=command_result.exit_code,
        working_directory=str(getattr(episode.sandbox, "merged_root", Path("/"))),
        execution_time=command_result.execution_time,
        reward=computation.signal.total_reward,
        done=done,
        step_number=episode.step_number,
        max_steps=episode.max_steps,
    )


def _merge_stderr(stderr: str, extra: str) -> str:
    if not stderr:
        return extra
    return f"{stderr.rstrip()}\n{extra}"


def _build_web_metadata() -> dict[str, Any]:
    return {
        "name": "sysadmin-env",
        "description": "Shell-based sysadmin environment with OpenEnv-compatible web shim routes.",
        "readme_content": _load_readme_content(),
        "documentation_url": "/docs",
    }


def _load_readme_content() -> str | None:
    readme_path = Path(__file__).resolve().parents[1] / "README.md"
    try:
        return readme_path.read_text(encoding="utf-8")
    except OSError:
        return None


def _build_web_step_result(result: StepResult) -> dict[str, Any]:
    observation = result.observation.model_dump()
    return {
        "observation": observation,
        "reward": result.observation.reward,
        "done": result.observation.done,
        "state": result.state.model_dump(),
    }


def _build_web_state(session: HttpSessionState) -> dict[str, Any]:
    if session.last_state is None:
        return {
            "episode_id": None,
            "task_id": None,
            "step_count": 0,
            "max_steps": 0,
            "done": False,
            "reward": 0.0,
            "initialized": False,
        }

    payload = session.last_state.model_dump()
    payload["initialized"] = True
    return payload


def _parse_web_step_request(payload: dict[str, Any]) -> StepRequest:
    action_payload = payload.get("action", payload)
    if not isinstance(action_payload, dict):
        raise HTTPException(status_code=422, detail="action payload must be an object")

    try:
        action = Action.model_validate(action_payload)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc

    return StepRequest(action=action)


def _render_web_interface_html() -> str:
    return """<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>sysadmin-env web shim</title>
  <style>
    body { font-family: system-ui, sans-serif; margin: 2rem auto; max-width: 960px; padding: 0 1rem; }
    h1, h2 { margin-bottom: 0.5rem; }
    .panel { border: 1px solid #d0d7de; border-radius: 8px; padding: 1rem; margin-bottom: 1rem; }
    .row { display: flex; gap: 0.75rem; flex-wrap: wrap; margin-bottom: 0.75rem; }
    input, select, button, textarea { font: inherit; padding: 0.5rem; }
    input, select, textarea { min-width: 240px; }
    textarea { width: 100%; min-height: 6rem; }
    pre { background: #0d1117; color: #e6edf3; padding: 1rem; overflow-x: auto; border-radius: 6px; }
    code { background: #f6f8fa; padding: 0.1rem 0.3rem; border-radius: 4px; }
  </style>
</head>
<body>
  <h1>sysadmin-env web compatibility shim</h1>
  <p>This page exposes the OpenEnv-compatible helper routes for the existing FastAPI environment without changing the primary HTTP or websocket API.</p>

  <div class=\"panel\">
    <h2>Reset</h2>
    <div class=\"row\">
      <select id=\"task-id\"></select>
      <button id=\"reset-button\" type=\"button\">POST /web/reset</button>
      <button id=\"state-button\" type=\"button\">GET /web/state</button>
      <button id=\"metadata-button\" type=\"button\">GET /web/metadata</button>
    </div>
  </div>

  <div class=\"panel\">
    <h2>Step</h2>
    <div class=\"row\">
      <input id=\"command\" type=\"text\" placeholder=\"echo hello\">
      <input id=\"reasoning\" type=\"text\" placeholder=\"optional reasoning\">
      <button id=\"step-button\" type=\"button\">POST /web/step</button>
    </div>
    <p>Route contract: <code>{\"action\": {\"command\": \"...\", \"reasoning\": \"...\"}}</code></p>
  </div>

  <div class=\"panel\">
    <h2>Response</h2>
    <pre id=\"output\">loading tasks...</pre>
  </div>

  <script>
    const output = document.getElementById('output');
    const taskSelect = document.getElementById('task-id');

    async function showResponse(response) {
      const text = await response.text();
      try {
        output.textContent = JSON.stringify(JSON.parse(text), null, 2);
      } catch {
        output.textContent = text;
      }
    }

    async function loadTasks() {
      const response = await fetch('/tasks');
      const payload = await response.json();
      taskSelect.innerHTML = payload.tasks.map((task) => `<option value="${task.task_id}">${task.task_id}</option>`).join('');
      output.textContent = JSON.stringify(payload, null, 2);
    }

    document.getElementById('reset-button').addEventListener('click', async () => {
      const response = await fetch('/web/reset', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ task_id: taskSelect.value || null }),
      });
      await showResponse(response);
    });

    document.getElementById('step-button').addEventListener('click', async () => {
      const payload = {
        action: {
          command: document.getElementById('command').value,
          reasoning: document.getElementById('reasoning').value || null,
        },
      };
      const response = await fetch('/web/step', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      await showResponse(response);
    });

    document.getElementById('state-button').addEventListener('click', async () => {
      const response = await fetch('/web/state');
      await showResponse(response);
    });

    document.getElementById('metadata-button').addEventListener('click', async () => {
      const response = await fetch('/web/metadata');
      await showResponse(response);
    });

    loadTasks().catch((error) => {
      output.textContent = `Failed to load tasks: ${error.message}`;
    });
  </script>
</body>
</html>
"""


def _build_environment_state(episode: EpisodeState, episode_id: str, observation: Observation) -> EnvironmentState:
    return EnvironmentState(
        episode_id=episode_id,
        task_id=episode.task_id,
        step_count=observation.step_number,
        max_steps=episode.max_steps,
        done=observation.done,
        reward=observation.reward,
    )


def _runtime_root_for_definition(sandbox: Sandbox, definition: TaskScenarioDefinition) -> Path:
    state_root = getattr(sandbox, "state_root", None)
    if state_root is not None:
        return Path(state_root)

    lowerdir = getattr(sandbox, "lowerdir", None)
    if lowerdir is not None:
        return Path(lowerdir)

    return Path(definition.metadata.base_filesystem_path)


def _synchronize_task_runtime(task_module, runtime_root: Path) -> None:
    synchronizer = getattr(task_module, "synchronize", None)
    if callable(synchronizer):
        synchronizer(runtime_root)


def _apply_task_runtime_updates(task_module, runtime_root: Path, command: str, command_result: CommandResult) -> None:
    observer = getattr(task_module, "observe_command", None)
    if callable(observer):
        observer(runtime_root, command, command_result)

    synchronizer = getattr(task_module, "synchronize", None)
    if callable(synchronizer):
        synchronizer(runtime_root)


async def _send_episode_started(websocket: WebSocket, manager: EpisodeManager, episode: EpisodeState) -> None:
    definition = manager.task_registry[episode.task_id]
    await websocket.send_json({
        "type": "episode_started",
        "task": {
            "task_id": definition.metadata.task_id,
            "difficulty": definition.metadata.difficulty.value,
            "description": definition.metadata.description,
            "max_steps": definition.metadata.max_steps,
            "time_limit": definition.metadata.time_limit,
        },
    })


async def _send_error(websocket: WebSocket, code: str, message: str) -> None:
    await websocket.send_json({
        "type": "error",
        "code": code,
        "message": message,
    })


app = create_app()
