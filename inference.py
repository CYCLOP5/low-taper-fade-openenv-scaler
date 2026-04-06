#!/usr/bin/env python3

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import websockets
from websockets.asyncio.client import ClientConnection


DEFAULT_SERVER_URL = "ws://127.0.0.1:8000/ws"
DEFAULT_HEALTHCHECK_URL = "http://127.0.0.1:8000/health"
DEFAULT_TASKS_URL = "http://127.0.0.1:8000/tasks"
DEFAULT_MODEL_API_URL = "https://api.openai.com/v1/responses"
DEFAULT_MODEL_NAME = "gpt-4o-mini"
DEFAULT_API_TIMEOUT = 20.0
DEFAULT_EPISODE_TIMEOUT = 600.0
MAX_REASONING_CHARS = 800


@dataclass
class AgentConfig:
    server_url: str
    healthcheck_url: str
    tasks_url: str
    model_api_url: str
    model_name: str
    reasoning_effort: str | None
    api_key: str | None
    api_timeout: float
    episode_timeout: float
    task_id: str | None


@dataclass
class ModelDecision:
    command: str
    reasoning: str | None
    source: str


def load_config() -> AgentConfig:
    _load_dotenv()
    return AgentConfig(
        server_url=os.getenv("SYSADMIN_ENV_SERVER_URL", DEFAULT_SERVER_URL),
        healthcheck_url=os.getenv("SYSADMIN_ENV_HEALTHCHECK_URL", DEFAULT_HEALTHCHECK_URL),
        tasks_url=os.getenv("SYSADMIN_ENV_TASKS_URL", DEFAULT_TASKS_URL),
        model_api_url=os.getenv("OPENAI_BASE_URL", DEFAULT_MODEL_API_URL),
        model_name=os.getenv("OPENAI_MODEL", os.getenv("MODEL_NAME", DEFAULT_MODEL_NAME)),
        reasoning_effort=_read_optional_env("OPENAI_REASONING_EFFORT") or _read_optional_env("REASONING_EFFORT"),
        api_key=os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY"),
        api_timeout=_parse_float_env("MODEL_API_TIMEOUT_SECONDS", DEFAULT_API_TIMEOUT),
        episode_timeout=_parse_float_env("EPISODE_TIMEOUT_SECONDS", DEFAULT_EPISODE_TIMEOUT),
        task_id=os.getenv("SYSADMIN_ENV_TASK_ID"),
    )


def _load_dotenv() -> None:
    dotenv_path = Path(os.getenv("SYSADMIN_ENV_DOTENV_PATH", Path(__file__).resolve().with_name(".env")))
    if not dotenv_path.is_file():
        return

    for raw_line in dotenv_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()

        if not key or key in os.environ:
            continue

        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]

        os.environ[key] = value


def _parse_float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _read_optional_env(name: str) -> str | None:
    value = os.getenv(name)
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        return None
    return stripped


async def run() -> int:
    config = load_config()
    try:
        await verify_server(config)
        task_sequence = await load_task_sequence(config)
        for task_id in task_sequence:
            print("[start]")
            try:
                await asyncio.wait_for(run_episode(config, task_id), timeout=config.episode_timeout)
            except asyncio.TimeoutError:
                print("step timeout")
                print("[end]")
                return 1
            except Exception as exc:
                print(_short_message(f"step failed {exc}"))
                print("[end]")
                return 1
            print("[end]")
    except KeyboardInterrupt:
        print("step interrupted")
        return 130
    except Exception as exc:
        print(_short_message(f"step failed {exc}"))
        return 1
    return 0


async def verify_server(config: AgentConfig) -> None:
    async with httpx.AsyncClient(timeout=config.api_timeout) as client:
        response = await client.get(config.healthcheck_url)
        response.raise_for_status()


async def load_task_sequence(config: AgentConfig) -> list[str]:
    if config.task_id:
        return [config.task_id]

    async with httpx.AsyncClient(timeout=config.api_timeout) as client:
        response = await client.get(config.tasks_url)
        response.raise_for_status()
        payload = response.json()

    task_items = payload.get("tasks", [])
    task_ids = [str(item.get("task_id", "")).strip() for item in task_items if item.get("task_id")]
    if task_ids:
        return task_ids

    return ["nginx_crash", "disk_full", "network_broken"]


async def run_episode(config: AgentConfig, task_id: str) -> None:
    websocket_url = _build_websocket_url(config, task_id)
    async with websockets.connect(websocket_url, open_timeout=config.api_timeout) as websocket:
        started = await _receive_json(websocket)
        if started.get("type") != "episode_started":
            raise RuntimeError(_extract_error_message(started))
        task = started["task"]
        print(_format_episode_started(task))
        history: list[dict[str, Any]] = []
        observation: dict[str, Any] | None = None

        while True:
            decision = await choose_action(config, task, observation, history)
            await websocket.send(json.dumps({
                "command": decision.command,
                "reasoning": decision.reasoning,
            }))
            message = await _receive_json(websocket)
            if message.get("type") == "error":
                raise RuntimeError(_extract_error_message(message))
            if message.get("type") != "observation":
                raise RuntimeError("unexpected websocket message")
            observation = message["observation"]
            history.append({
                "action": decision.command,
                "reasoning": decision.reasoning,
                "source": decision.source,
                "observation": observation,
            })
            print(_format_step(decision, observation))
            if observation.get("done"):
                print(_short_message(f"episode reward {observation.get('reward', 0.0)}"))
                return


def _build_websocket_url(config: AgentConfig, task_id: str) -> str:
    separator = "&" if "?" in config.server_url else "?"
    return f"{config.server_url}{separator}task_id={task_id}"


async def choose_action(
    config: AgentConfig,
    task: dict[str, Any],
    observation: dict[str, Any] | None,
    history: list[dict[str, Any]],
) -> ModelDecision:
    if config.api_key:
        decision = await request_model_action(config, task, observation, history)
        if decision is not None:
            return decision
    return heuristic_action(task, observation, history)


async def request_model_action(
    config: AgentConfig,
    task: dict[str, Any],
    observation: dict[str, Any] | None,
    history: list[dict[str, Any]],
) -> ModelDecision | None:
    payload = _build_model_request_payload(config, task, observation, history)
    headers = {
        "authorization": f"Bearer {config.api_key}",
        "content-type": "application/json",
    }
    try:
        async with httpx.AsyncClient(timeout=config.api_timeout) as client:
            response = await client.post(config.model_api_url, headers=headers, json=payload)
            if response.status_code in {401, 403, 404, 408, 429, 500, 502, 503, 504}:
                print(_short_message(f"step api {response.status_code}"))
                return None
            response.raise_for_status()
            data = response.json()
    except (httpx.TimeoutException, httpx.NetworkError):
        print("step api timeout")
        return None
    except httpx.HTTPError as exc:
        print(_short_message(f"step api error {exc}"))
        return None
    except ValueError:
        print("step api invalid")
        return None

    if data.get("status") == "incomplete":
        incomplete = data.get("incomplete_details", {})
        reason = incomplete.get("reason")
        if isinstance(reason, str):
            print(_short_message(f"step api incomplete {reason}"))

    content = _extract_model_content(data)
    if content is None:
        print("step api empty")
        return None
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        print("step api json")
        return None
    command = str(parsed.get("command", "")).strip()
    if not command:
        print("step api command")
        return None
    reasoning = parsed.get("reasoning")
    if reasoning is not None:
        reasoning = _short_message(str(reasoning), MAX_REASONING_CHARS)
    return ModelDecision(command=command, reasoning=reasoning, source="model")


def _build_model_request_payload(
    config: AgentConfig,
    task: dict[str, Any],
    observation: dict[str, Any] | None,
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    system_prompt = (
        "you are a linux remediation agent "
        "return strict json with command and reasoning "
        "choose one safe shell command per turn"
    )
    user_payload = json.dumps({
        "task": task,
        "last_observation": observation,
        "history": history[-6:],
        "constraints": {
            "single_command": True,
            "avoid_destructive_actions": True,
        },
    })

    if _uses_responses_api(config.model_api_url):
        payload = {
            "model": config.model_name,
            "input": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_payload,
                },
            ],
        }
        if config.reasoning_effort is not None:
            payload["reasoning"] = {"effort": config.reasoning_effort}
        return payload

    payload = {
        "model": config.model_name,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_payload,
            },
        ],
    }
    return payload


def _uses_responses_api(model_api_url: str) -> bool:
    return model_api_url.rstrip("/").endswith("/responses")


def _extract_model_content(data: dict[str, Any]) -> str | None:
    output_text = data.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text

    output = data.get("output")
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict) or item.get("type") != "message":
                continue
            content_items = item.get("content", [])
            if not isinstance(content_items, list):
                continue
            for content_item in content_items:
                if not isinstance(content_item, dict):
                    continue
                text = content_item.get("text")
                if isinstance(text, str) and text.strip():
                    return text

    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    message = choices[0].get("message", {})
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text")
                if isinstance(text, str):
                    return text
    return None


def heuristic_action(
    task: dict[str, Any],
    observation: dict[str, Any] | None,
    history: list[dict[str, Any]],
) -> ModelDecision:
    task_id = str(task.get("task_id", ""))
    attempts = len(history)
    command = _task_plan(task_id, observation, attempts)
    return ModelDecision(command=command, reasoning="fallback heuristic", source="fallback")


def _task_plan(task_id: str, observation: dict[str, Any] | None, attempts: int) -> str:
    if task_id == "nginx_crash":
        plan = [
            "cat /var/log/nginx/error.log",
            "cat /var/run/nginx.pid",
            "rm -f /var/run/nginx.pid",
            "nginx -t",
            "sed -i 's/listen 8080$/listen 8080;/' /etc/nginx/nginx.conf",
            "nginx -t",
            "nginx",
            "curl -I http://127.0.0.1:8080",
        ]
        return plan[min(attempts, len(plan) - 1)]

    if task_id == "disk_full":
        plan = [
            "df -h /mnt/data",
            "du -sh /mnt/data /mnt/data/.cache /mnt/data/.cache/.rotated 2>/dev/null",
            "find /mnt/data -type f | sort",
            "ls -lh /mnt/data/.cache/.rotated/app.trace",
            "truncate -s 0 /mnt/data/.cache/.rotated/app.trace",
            "df -h /mnt/data",
        ]
        return plan[min(attempts, len(plan) - 1)]

    if task_id == "network_broken":
        plan = [
            "ip route show",
            "ip addr",
            "cat /etc/resolv.conf",
            "printf 'default via 10.0.2.2 dev eth0\n' > /etc/network/routes/default",
            "printf 'nameserver 1.1.1.1\n' > /etc/resolv.conf",
            "ping -c 1 example.com",
        ]
        return plan[min(attempts, len(plan) - 1)]

    generic_plan = [
        "pwd",
        "ls -la",
        "find . -maxdepth 3 -type f | sort | head -50",
        "env | sort",
    ]
    return generic_plan[min(attempts, len(generic_plan) - 1)]


async def _receive_json(websocket: ClientConnection) -> dict[str, Any]:
    raw_message = await websocket.recv()
    if not isinstance(raw_message, str):
        raise RuntimeError("unexpected websocket payload")
    try:
        return json.loads(raw_message)
    except json.JSONDecodeError as exc:
        raise RuntimeError("invalid websocket json") from exc


def _extract_error_message(message: dict[str, Any]) -> str:
    code = message.get("code", "unknown")
    detail = message.get("message", "unknown error")
    return f"{code} {detail}"


def _format_episode_started(task: dict[str, Any]) -> str:
    payload = {
        "task_id": task.get("task_id"),
        "difficulty": task.get("difficulty"),
        "max_steps": task.get("max_steps"),
        "time_limit": task.get("time_limit"),
        "description": task.get("description"),
    }
    return f"[step] {json.dumps({'event': 'episode_started', 'task': payload}, ensure_ascii=False)}"


def _format_step(decision: ModelDecision, observation: dict[str, Any]) -> str:
    payload = {
        "action": {
            "command": decision.command,
            "reasoning": decision.reasoning,
            "source": decision.source,
        },
        "observation": {
            "stdout": observation.get("stdout", ""),
            "stderr": observation.get("stderr", ""),
            "exit_code": observation.get("exit_code"),
            "working_directory": observation.get("working_directory"),
            "execution_time": observation.get("execution_time"),
            "reward": observation.get("reward"),
            "done": observation.get("done"),
            "step_number": observation.get("step_number"),
            "max_steps": observation.get("max_steps"),
        },
    }
    return f"[step] {json.dumps(payload, ensure_ascii=False)}"


def _short_message(value: str, limit: int = 120) -> str:
    compact = " ".join(value.strip().split())
    if len(compact) <= limit:
        return compact.lower()
    return compact[: limit - 3].lower() + "..."


def main() -> None:
    raise SystemExit(asyncio.run(run()))


if __name__ == "__main__":
    main()
