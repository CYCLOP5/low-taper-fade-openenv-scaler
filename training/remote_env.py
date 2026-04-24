from __future__ import annotations

import itertools
import random
import threading
import time
from dataclasses import dataclass
from typing import Any
from typing import Iterable

try:
    import httpx
except ImportError as exc:
    raise ImportError("httpx is required for training.remote_env install httpx") from exc

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:
    raise ImportError("gymnasium is required for training.remote_env") from exc


@dataclass
class RemoteStep:
    stdout: str
    stderr: str
    exit_code: int
    reward: float
    done: bool
    step_number: int
    max_steps: int
    task_id: str
    episode_id: str


class RemoteEndpointPool:
    def __init__(
        self,
        env_urls: Iterable[str],
        timeout: float = 60.0,
        retries: int = 3,
        backoff_seconds: float = 1.5,
        api_key: str | None = None,
    ) -> None:
        cleaned = [u.rstrip("/") for u in env_urls if u.strip()]
        if not cleaned:
            raise ValueError("at least one env_url is required")
        self._urls = cleaned
        self._cycle = itertools.cycle(self._urls)
        self._lock = threading.Lock()
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.Client(timeout=timeout, http2=False, headers=headers)
        self._retries = retries
        self._backoff = backoff_seconds

    def next_url(self) -> str:
        with self._lock:
            return next(self._cycle)

    def post_json(self, url: str, path: str, body: dict[str, Any] | None) -> dict[str, Any]:
        last_exc: Exception | None = None
        for attempt in range(self._retries):
            try:
                resp = self._client.post(f"{url}{path}", json=body or {})
                resp.raise_for_status()
                return resp.json()
            except Exception as exc:
                last_exc = exc
                time.sleep(self._backoff * (attempt + 1))
        raise RuntimeError(f"remote call failed {url}{path}") from last_exc

    def get_json(self, url: str, path: str) -> dict[str, Any]:
        resp = self._client.get(f"{url}{path}")
        resp.raise_for_status()
        return resp.json()

    def close(self) -> None:
        self._client.close()


class HttpEnterpriseHPCEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        env_urls: list[str] | str,
        *,
        scenario: str | None = None,
        scenario_pool: list[str] | None = None,
        pool: RemoteEndpointPool | None = None,
        timeout: float = 60.0,
        api_key: str | None = None,
    ) -> None:
        super().__init__()
        self.action_space = spaces.Text(max_length=4096)
        self.observation_space = spaces.Text(max_length=65536)

        _external_pool = pool is not None
        if pool is None:
            if isinstance(env_urls, str):
                env_urls = [env_urls]
            pool = RemoteEndpointPool(env_urls, timeout=timeout, api_key=api_key)
        self._pool = pool
        self._owns_pool = not _external_pool
        self._scenario = scenario
        self._scenario_pool = scenario_pool or ([scenario] if scenario else ["hpc_outage", "hpc_munge"])
        self._active_url: str | None = None
        self._episode_task: str | None = None
        self._episode_id: str | None = None
        self._step_count = 0
        self._max_steps = 0
        self._rng = random.Random()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng.seed(seed)

        task_id = None
        if options and "scenario" in options:
            task_id = options["scenario"]
        elif self._scenario:
            task_id = self._scenario
        elif self._scenario_pool:
            task_id = self._rng.choice(self._scenario_pool)

        self._active_url = self._pool.next_url()
        body = {"task_id": task_id} if task_id else {}
        data = self._pool.post_json(self._active_url, "/reset", body)
        self._absorb(data)

        obs_payload = data["observation"]
        observation = self._observation_text(obs_payload)
        info = {
            "task_id": self._episode_task or "",
            "episode_id": self._episode_id or "",
            "max_steps": self._max_steps,
            "endpoint": self._active_url,
            "grader_health": float(obs_payload.get("grader_health", 0.0)),
            "grader_details": dict(obs_payload.get("grader_details") or {}),
            "ood_http_code": str(obs_payload.get("ood_http_code", "")),
        }
        return observation, info

    def step(
        self, action: str
    ) -> tuple[str, float, bool, bool, dict[str, Any]]:
        if self._active_url is None:
            raise RuntimeError("HttpEnterpriseHPCEnv step called before reset")

        body: dict[str, Any] = {
            "action": {"command": action, "reasoning": None},
        }
        # include episode_id so the server-side multi-session store routes
        # this step to the correct episode. older servers ignore the field.
        if self._episode_id:
            body["episode_id"] = self._episode_id

        data = self._pool.post_json(self._active_url, "/step", body)
        self._absorb(data)
        obs_payload = data["observation"]
        obs = self._observation_text(obs_payload)
        reward = float(obs_payload.get("reward", 0.0))
        terminated = bool(obs_payload.get("done", False))
        truncated = (not terminated) and self._step_count >= self._max_steps

        info = {
            "task_id": self._episode_task or "",
            "episode_id": self._episode_id or "",
            "step": self._step_count,
            "max_steps": self._max_steps,
            "exit_code": obs_payload.get("exit_code"),
            "reward_source": "openenv_remote",
            "endpoint": self._active_url,
            "grader_health": float(obs_payload.get("grader_health", 0.0)),
            "grader_details": dict(obs_payload.get("grader_details") or {}),
            "ood_http_code": str(obs_payload.get("ood_http_code", "")),
        }
        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        if self._owns_pool:
            self._pool.close()

    def _absorb(self, data: dict[str, Any]) -> None:
        state = data.get("state") or {}
        self._episode_task = state.get("task_id")
        self._episode_id = state.get("episode_id")
        self._step_count = int(state.get("step_count", 0))
        self._max_steps = int(state.get("max_steps", 1))

    @staticmethod
    def _observation_text(obs: dict[str, Any]) -> str:
        stdout = obs.get("stdout", "")
        stderr = obs.get("stderr", "")
        parts: list[str] = []
        if stdout:
            parts.append(stdout.rstrip())
        if stderr:
            parts.append("stderr:\n" + stderr.rstrip())
        if not parts:
            parts.append("[command executed no output]")
        return "\n".join(parts)
