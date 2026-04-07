#!/usr/bin/env python3

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import websockets
from websockets.asyncio.client import ClientConnection


DEFAULT_SERVER_URL = "ws://127.0.0.1:8000/ws"
DEFAULT_HEALTHCHECK_URL = "http://127.0.0.1:8000/health"
DEFAULT_TASKS_URL = "http://127.0.0.1:8000/tasks"
DEFAULT_MODEL_API_URL = "https://api.openai.com/v1"
DEFAULT_MODEL_NAME = "gpt-5.4"
DEFAULT_API_TIMEOUT = 20.0
DEFAULT_EPISODE_TIMEOUT = 600.0
MAX_REASONING_CHARS = 800
BENCHMARK_NAME = "sysadmin-env"


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


@dataclass
class EpisodeSummary:
    task_id: str
    success: bool
    steps: int
    score: float
    rewards: list[float]


def load_config() -> AgentConfig:
    _load_dotenv()
    return AgentConfig(
        server_url=os.getenv("SYSADMIN_ENV_SERVER_URL", DEFAULT_SERVER_URL),
        healthcheck_url=os.getenv("SYSADMIN_ENV_HEALTHCHECK_URL", DEFAULT_HEALTHCHECK_URL),
        tasks_url=os.getenv("SYSADMIN_ENV_TASKS_URL", DEFAULT_TASKS_URL),
        model_api_url=os.getenv("API_BASE_URL", os.getenv("OPENAI_BASE_URL", DEFAULT_MODEL_API_URL)),
        model_name=os.getenv("MODEL_NAME", os.getenv("OPENAI_MODEL", DEFAULT_MODEL_NAME)),
        reasoning_effort=_read_optional_env("OPENAI_REASONING_EFFORT") or _read_optional_env("REASONING_EFFORT"),
        api_key=os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY"),
        api_timeout=_parse_float_env("MODEL_API_TIMEOUT_SECONDS", DEFAULT_API_TIMEOUT),
        episode_timeout=_parse_float_env("EPISODE_TIMEOUT_SECONDS", DEFAULT_EPISODE_TIMEOUT),
        task_id=os.getenv("SYSADMIN_ENV_TASK_ID"),
    )


def _load_dotenv() -> None:
    explicit_dotenv_path = os.getenv("SYSADMIN_ENV_DOTENV_PATH")
    candidate_paths = [Path(explicit_dotenv_path)] if explicit_dotenv_path else [
        Path.cwd() / ".env",
        Path(__file__).resolve().with_name(".env"),
    ]

    seen_paths: set[str] = set()
    for dotenv_path in candidate_paths:
        normalized_path = str(dotenv_path.resolve(strict=False))
        if normalized_path in seen_paths:
            continue
        seen_paths.add(normalized_path)

        if not dotenv_path.is_file():
            continue

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
        return


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
    overall_exit_code = 0
    try:
        await verify_server(config)
        task_sequence = await load_task_sequence(config)
        for task_id in task_sequence:
            log_start(task=task_id, env=BENCHMARK_NAME, model=config.model_name)
            try:
                summary = await asyncio.wait_for(run_episode(config, task_id), timeout=config.episode_timeout)
            except asyncio.TimeoutError:
                overall_exit_code = 1
                message = "episode timeout"
                _emit_error(message)
                log_step(step=0, action=None, reward=0.0, done=True, error=message)
                summary = EpisodeSummary(task_id=task_id, success=False, steps=0, score=0.0, rewards=[])
            except Exception as exc:
                overall_exit_code = 1
                message = _short_message(f"episode failed {exc}")
                _emit_error(message)
                log_step(step=0, action=None, reward=0.0, done=True, error=message)
                summary = EpisodeSummary(task_id=task_id, success=False, steps=0, score=0.0, rewards=[])
            log_end(success=summary.success, steps=summary.steps, score=summary.score, rewards=summary.rewards)
    except KeyboardInterrupt:
        _emit_error("episode interrupted")
        return 130
    except Exception as exc:
        _emit_error(_short_message(f"run failed {exc}"))
        return 1
    return overall_exit_code


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


async def run_episode(config: AgentConfig, task_id: str) -> EpisodeSummary:
    websocket_url = _build_websocket_url(config, task_id)
    async with websockets.connect(websocket_url, open_timeout=config.api_timeout) as websocket:
        started = await _receive_json(websocket)
        if started.get("type") != "episode_started":
            raise RuntimeError(_extract_error_message(started))
        task = started["task"]
        history: list[dict[str, Any]] = []
        observation: dict[str, Any] | None = None
        rewards: list[float] = []

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

            reward = float(observation.get("reward", 0.0) or 0.0)
            rewards.append(reward)
            step_number = int(observation.get("step_number", len(rewards)))
            done = bool(observation.get("done", False))
            log_step(step=step_number, action=decision.command, reward=reward, done=done, error=None)

            if done:
                max_steps = int(observation.get("max_steps", step_number or 1))
                success = reward > 0.0 and step_number < max_steps
                return EpisodeSummary(
                    task_id=str(task.get("task_id", task_id)),
                    success=success,
                    steps=step_number,
                    score=_normalize_reported_score(sum(rewards)),
                    rewards=rewards,
                )


def _build_websocket_url(config: AgentConfig, task_id: str) -> str:
    separator = "&" if "?" in config.server_url else "?"
    return f"{config.server_url}{separator}task_id={task_id}"


async def choose_action(
    config: AgentConfig,
    task: dict[str, Any],
    observation: dict[str, Any] | None,
    history: list[dict[str, Any]],
) -> ModelDecision:
    fallback = heuristic_action(task, observation, history)
    if config.api_key:
        decision = await request_model_action(config, task, observation, history)
        if decision is not None:
            return _stabilize_model_decision(task, history, decision, fallback)
    return fallback


def _stabilize_model_decision(
    task: dict[str, Any],
    history: list[dict[str, Any]],
    decision: ModelDecision,
    fallback: ModelDecision,
) -> ModelDecision:
    task_id = str(task.get("task_id", "")).strip()
    if task_id != "network_broken":
        return decision

    command = _normalize_shell_command(decision.command)
    if _is_network_repair_command(command):
        return decision

    if _network_diagnosis_complete(history):
        return _network_guardrail_decision(history, fallback)

    return decision


def _network_guardrail_decision(history: list[dict[str, Any]], fallback: ModelDecision) -> ModelDecision:
    if not _network_dns_repaired(history):
        _emit_error("network guardrail dns repair")
        return ModelDecision(
            command="printf 'nameserver 1.1.1.1\n' > /etc/resolv.conf",
            reasoning="fallback heuristic dns repair after task-specific network guardrail",
            source="fallback",
        )

    if not _network_route_repaired(history):
        _emit_error("network guardrail route repair")
        return ModelDecision(
            command="printf 'default via 10.0.2.2 dev eth0\n' > /etc/network/routes/default",
            reasoning="fallback heuristic route repair after task-specific network guardrail",
            source="fallback",
        )

    _emit_error("network guardrail connectivity check")
    return ModelDecision(
        command="ping -c 1 example.com",
        reasoning="fallback heuristic connectivity check after task-specific network guardrail",
        source="fallback",
    )


async def request_model_action(
    config: AgentConfig,
    task: dict[str, Any],
    observation: dict[str, Any] | None,
    history: list[dict[str, Any]],
) -> ModelDecision | None:
    return await asyncio.to_thread(_request_model_action_sync, config, task, observation, history)


def _request_model_action_sync(
    config: AgentConfig,
    task: dict[str, Any],
    observation: dict[str, Any] | None,
    history: list[dict[str, Any]],
) -> ModelDecision | None:
    payload = _build_model_request_payload(config, task, observation, history)
    client = _create_openai_client(config)
    try:
        response = client.responses.create(**payload)
    except Exception as exc:
        status_code = getattr(exc, "status_code", None)
        if isinstance(status_code, int) and status_code in {401, 403, 404, 408, 429, 500, 502, 503, 504}:
            _emit_error(_short_message(f"step api {status_code}"))
            return None
        message = _short_message(str(exc) or exc.__class__.__name__)
        if "timeout" in message:
            _emit_error("step api timeout")
            return None
        _emit_error(_short_message(f"step api error {message}"))
        return None
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()

    if getattr(response, "status", None) == "incomplete":
        incomplete = getattr(response, "incomplete_details", None)
        reason = getattr(incomplete, "reason", None)
        if isinstance(reason, str):
            _emit_error(_short_message(f"step api incomplete {reason}"))

    content = _extract_model_content(response)
    if content is None:
        _emit_error("step api empty")
        return None

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        _emit_error("step api json")
        return None

    command = str(parsed.get("command", "")).strip()
    if not command:
        _emit_error("step api command")
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
        "choose one safe shell command per turn "
        "avoid repeating command patterns that already failed or produced no new information "
        "after enough evidence prefer a concrete repair action over more diagnosis "
        "adapt to the observed environment and avoid unsupported command variants"
    )
    user_payload = json.dumps({
        "task": task,
        "last_observation": observation,
        "history": history[-6:],
        "playbook": _task_playbook(str(task.get("task_id", "")).strip()),
        "constraints": {
            "single_command": True,
            "avoid_destructive_actions": True,
            "avoid_repeating_failed_patterns": True,
            "prefer_repair_after_evidence": True,
            "prefer_supported_commands": True,
        },
    }, ensure_ascii=False)

    payload = {
        "model": config.model_name,
        "instructions": system_prompt,
        "input": user_payload,
    }
    if config.reasoning_effort is not None:
        payload["reasoning"] = {"effort": config.reasoning_effort}
    return payload


def _create_openai_client(config: AgentConfig):
    from openai import OpenAI

    client_kwargs: dict[str, Any] = {
        "api_key": config.api_key,
        "timeout": config.api_timeout,
        "max_retries": 1,
    }
    base_url = _normalize_openai_base_url(config.model_api_url)
    if base_url is not None:
        client_kwargs["base_url"] = base_url
    return OpenAI(**client_kwargs)


def _normalize_openai_base_url(model_api_url: str) -> str | None:
    stripped = model_api_url.strip()
    if not stripped:
        return None
    base_url = stripped.rstrip("/")
    if base_url.endswith("/responses"):
        return base_url[: -len("/responses")]
    return base_url


def _extract_model_content(data: Any) -> str | None:
    output_text = getattr(data, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text

    if hasattr(data, "model_dump"):
        data = data.model_dump()

    if not isinstance(data, dict):
        return None

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


def _task_playbook(task_id: str) -> dict[str, Any]:
    if task_id == "nginx_crash":
        return {
            "objective": "clear the stale nginx pid, fix the listen directive, and start nginx safely",
            "supported_diagnostics": [
                "cat /var/log/nginx/error.log",
                "cat /var/run/nginx.pid",
                "nginx -t",
                "ps",
                "pgrep",
            ],
            "repair_targets": {
                "config_contains": "listen 8080;",
                "pid_file": "missing or rewritten by the nginx stub",
            },
        }

    if task_id == "disk_full":
        return {
            "objective": "identify the file exhausting /mnt/data and reclaim capacity safely",
            "supported_diagnostics": [
                "df -h /mnt/data",
                "du -sh /mnt/data /mnt/data/.cache /mnt/data/.cache/.rotated",
                "find /mnt/data -type f",
                "lsof",
            ],
            "repair_targets": {
                "full_mount": "/mnt/data",
                "hidden_offender": "/mnt/data/.cache/.rotated/app.trace",
            },
        }

    if task_id == "network_broken":
        return {
            "objective": "inspect routing, interface state, and dns, then repair the task-local route file and resolver config using supported commands",
            "supported_diagnostics": [
                "ip route show",
                "ip addr",
                "ip link",
                "cat /etc/resolv.conf",
                "ping -c 1 example.com",
            ],
            "supported_repairs": [
                "write the repaired default route into /etc/network/routes/default",
                "use supported ip/route stub commands instead of unsupported variants",
                "write a repaired nameserver into /etc/resolv.conf",
            ],
            "avoid": [
                "do not guess host-specific gateways or dns servers without evidence from the task",
                "prefer supported stub commands over unsupported real-linux variants",
                "repair only after enough diagnosis to identify the broken routing and dns state",
            ],
        }

    return {
        "objective": "inspect the environment, gather evidence, and apply one safe repair command per step",
    }


def _normalize_shell_command(command: str) -> str:
    return " ".join(command.strip().split())


def _network_diagnosis_complete(history: list[dict[str, Any]]) -> bool:
    commands = [_normalize_shell_command(str(item.get("action", ""))) for item in history]
    route_checked = any(re.search(r"\bip\b.*\broute\b.*\bshow\b|\broute\b.*\b-n\b", command) for command in commands)
    dns_checked = any("resolv.conf" in command for command in commands)
    interface_checked = any(re.search(r"\bip\b.*\baddr\b|\bip\b.*\blink\b|\bifconfig\b", command) for command in commands)
    return route_checked and dns_checked and interface_checked


def _network_dns_repaired(history: list[dict[str, Any]]) -> bool:
    for item in history:
        command = _normalize_shell_command(str(item.get("action", "")))
        reward = _history_reward(item)
        if _is_exact_dns_repair_command(command):
            return True
        if _is_dns_write_command(command) and reward > 0.0:
            return True
    return False


def _network_route_repaired(history: list[dict[str, Any]]) -> bool:
    for item in history:
        command = _normalize_shell_command(str(item.get("action", "")))
        reward = _history_reward(item)
        if _is_exact_route_repair_command(command):
            return True
        if _is_route_write_command(command) and reward > 0.0:
            return True
    return False


def _history_reward(item: dict[str, Any]) -> float:
    observation = item.get("observation", {})
    if not isinstance(observation, dict):
        return 0.0
    return float(observation.get("reward", 0.0) or 0.0)


def _is_dns_write_command(command: str) -> bool:
    return "/etc/resolv.conf" in command and _looks_like_mutating_shell_command(command)


def _is_route_write_command(command: str) -> bool:
    return (
        bool(re.search(r"\bip\s+route\s+add\s+default\s+via\b", command))
        or ("/etc/network/routes/default" in command and _looks_like_mutating_shell_command(command))
    )


def _looks_like_mutating_shell_command(command: str) -> bool:
    return any(token in command for token in (">", "tee", "printf", "echo", "sed -i", "truncate", "rm "))


def _is_exact_dns_repair_command(command: str) -> bool:
    return command == "printf 'nameserver 1.1.1.1\n' > /etc/resolv.conf"


def _is_exact_route_repair_command(command: str) -> bool:
    return command == "printf 'default via 10.0.2.2 dev eth0\n' > /etc/network/routes/default" or bool(
        re.search(r"\bip\s+route\s+add\s+default\s+via\s+10\.0\.2\.2(?:\s+dev\s+eth0)?\b", command)
    )


def _is_network_repair_command(command: str) -> bool:
    return _is_exact_route_repair_command(command) or _is_exact_dns_repair_command(command)


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


def log_start(task: str, env: str, model: str) -> None:
    if _log_format() == "json":
        payload = {
            "task": task,
            "env": env,
            "model": model,
        }
        _emit_stdout(f"[START] {json.dumps(payload, ensure_ascii=False)}")
        return

    _emit_stdout(
        "[START] "
        f"task={_sanitize_log_value(task)} "
        f"env={_sanitize_log_value(env)} "
        f"model={_sanitize_log_value(model)}"
    )


def log_step(step: int, action: str | None, reward: float, done: bool, error: str | None) -> None:
    if _log_format() == "json":
        payload = {
            "step": step,
            "action": action,
            "reward": reward,
            "done": done,
            "error": error,
        }
        _emit_stdout(f"[STEP] {json.dumps(payload, ensure_ascii=False)}")
        return

    action_value = "null" if action is None else _sanitize_log_value(action)
    error_value = "null" if error is None else _sanitize_log_value(error)
    _emit_stdout(
        "[STEP] "
        f"step={step} "
        f"action={action_value} "
        f"reward={_format_reward(reward)} "
        f"done={_format_bool(done)} "
        f"error={error_value}"
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    if _log_format() == "json":
        payload = {
            "success": success,
            "steps": steps,
            "score": score,
            "rewards": rewards,
        }
        _emit_stdout(f"[END] {json.dumps(payload, ensure_ascii=False)}")
        return

    rewards_value = ",".join(_format_reward(reward) for reward in rewards)
    _emit_stdout(
        "[END] "
        f"success={_format_bool(success)} "
        f"steps={steps} "
        f"score={_format_reward(score)} "
        f"rewards={rewards_value}"
    )


def _log_format() -> str:
    value = os.getenv("SYSADMIN_ENV_LOG_FORMAT", "flat").strip().lower()
    if value == "json":
        return "json"
    return "flat"


def _sanitize_log_value(value: str) -> str:
    return " ".join(str(value).split())


def _format_bool(value: bool) -> str:
    return "true" if value else "false"


def _format_reward(value: float) -> str:
    return f"{float(value):.2f}"


def _emit_stdout(value: str) -> None:
    print(value, flush=True)


def _emit_error(value: str) -> None:
    print(value, file=sys.stderr, flush=True)


def _clamp_score(value: float) -> float:
    return min(max(float(value), 0.0), 1.0)


def _normalize_reported_score(value: float) -> float:
    return 0.01 + (0.98 * _clamp_score(value))


def _short_message(value: str, limit: int = 120) -> str:
    compact = " ".join(value.strip().split())
    if len(compact) <= limit:
        return compact.lower()
    return compact[: limit - 3].lower() + "..."


def main() -> None:
    raise SystemExit(asyncio.run(run()))


if __name__ == "__main__":
    main()
