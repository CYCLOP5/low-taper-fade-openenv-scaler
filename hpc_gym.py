from __future__ import annotations

import os
import random
import re
import shutil
import tempfile
import time
from pathlib import Path
from types import ModuleType
from typing import Any
from typing import Sequence

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:
    raise ImportError(
        "gymnasium is required for hpc_gym import with pip install gymnasium"
    ) from exc

try:
    import pexpect
except ImportError as exc:
    raise ImportError(
        "pexpect is required for hpc_gym import with pip install pexpect"
    ) from exc

from sysadmin_env.sandbox import Sandbox
from sysadmin_env.tasks import hpc_gpu_ecc
from sysadmin_env.tasks import hpc_munge
from sysadmin_env.tasks import hpc_nfs_stale
from sysadmin_env.tasks import hpc_ood_apache
from sysadmin_env.tasks import hpc_outage
from sysadmin_env.tasks import hpc_pid_stale


PROMPT_PATTERN = re.compile(r"\[[^\]\r\n]+\][#$]\s?")
PRIMARY_HOSTNAME = "hpc-login"
OOD_PORT = 8080
OOD_LOG_PATH = "/tmp/ood.log"
OOD_DAEMON_SCRIPT = "/usr/local/bin/ood_server.py"
DEFAULT_STEP_TIMEOUT = 60.0
DEFAULT_SHELL_TIMEOUT = 30.0

SCENARIO_REGISTRY: dict[str, ModuleType] = {
    hpc_outage.TASK_ID: hpc_outage,
    hpc_munge.TASK_ID: hpc_munge,
    hpc_pid_stale.TASK_ID: hpc_pid_stale,
    hpc_gpu_ecc.TASK_ID: hpc_gpu_ecc,
    hpc_nfs_stale.TASK_ID: hpc_nfs_stale,
    hpc_ood_apache.TASK_ID: hpc_ood_apache,
}


def resolve_scenario(name_or_module: str | ModuleType) -> ModuleType:
    if isinstance(name_or_module, ModuleType):
        return name_or_module
    if name_or_module in SCENARIO_REGISTRY:
        return SCENARIO_REGISTRY[name_or_module]
    raise KeyError(
        f"unknown scenario {name_or_module} expected one of {sorted(SCENARIO_REGISTRY)}"
    )


class EnterpriseHPCEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        task_root: str | None = None,
        *,
        scenario: str | ModuleType = hpc_outage.TASK_ID,
        scenario_pool: Sequence[str | ModuleType] | None = None,
        overlay_base_dir: str | None = None,
        shell_timeout: float = DEFAULT_SHELL_TIMEOUT,
        step_timeout: float = DEFAULT_STEP_TIMEOUT,
    ) -> None:
        super().__init__()
        self.action_space = spaces.Text(max_length=4096)
        self.observation_space = spaces.Text(max_length=65536)

        self._configured_task_root = task_root
        self._overlay_base_dir = overlay_base_dir
        self._shell_timeout = shell_timeout
        self._step_timeout = step_timeout

        self._scenario_pool: list[ModuleType]
        if scenario_pool is not None:
            self._scenario_pool = [resolve_scenario(item) for item in scenario_pool]
        else:
            self._scenario_pool = [resolve_scenario(scenario)]

        self._scenario: ModuleType = self._scenario_pool[0]
        self._sandbox: Sandbox | None = None
        self._sandbox_scenario_id: str | None = None
        self._shell: pexpect.spawn | None = None
        self._tmp_task_dir: str | None = None
        self._step_count = 0
        self._max_steps = 0
        self._last_reward = 0.0
        self._ood_started = False
        self._rng = random.Random()
        self._prev_health = 0.0

    @property
    def sandbox(self) -> Sandbox | None:
        return self._sandbox

    @property
    def scenario(self) -> ModuleType:
        return self._scenario

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng.seed(seed)

        self._select_scenario(options)
        self._close_shell()

        scenario_changed = self._sandbox_scenario_id != self._scenario.TASK_ID
        if self._sandbox is not None and scenario_changed:
            try:
                self._sandbox.destroy()
            except Exception as exc:
                print(f"hpc_gym sandbox destroy failed {type(exc).__name__.lower()} {exc}")
            self._sandbox = None

        task_root = self._ensure_task_root()
        if self._sandbox is None:
            print(f"hpc_gym create sandbox scenario {self._scenario.TASK_ID} task_root {task_root}")
            self._sandbox = Sandbox(
                task_root,
                timeout=self._step_timeout,
                isolate_network=False,
                overlay_base_dir=self._overlay_base_dir,
                allow_nested_sandbox=True,
            )
            self._sandbox.create()
            self._sandbox_scenario_id = self._scenario.TASK_ID
        else:
            start = time.perf_counter()
            latency_ms = self._sandbox.reset()
            print(
                f"hpc_gym overlay reset scenario {self._scenario.TASK_ID} "
                f"{latency_ms:.2f}ms wall {((time.perf_counter()-start)*1000):.2f}ms"
            )

        if self._sandbox.state_root is not None:
            self._scenario.synchronize(self._sandbox.state_root)

        definition = self._scenario.build_definition(str(self._sandbox.state_root or ""))
        self._max_steps = definition.metadata.max_steps
        self._step_count = 0
        self._last_reward = 0.0
        self._prev_health = 0.0
        self._ood_started = False

        self._spawn_shell()
        self._bootstrap_primary_prompt()
        self._launch_ood_daemon()
        self._enter_login_node()

        observation = (
            f"login node ready scenario {self._scenario.TASK_ID} ood :"
            f"{OOD_PORT} max_steps {self._max_steps}"
        )
        info = {
            "task_id": self._scenario.TASK_ID,
            "max_steps": self._max_steps,
            "ood_port": OOD_PORT,
            "prompt_pattern": PROMPT_PATTERN.pattern,
        }
        return observation, info

    def step(
        self, action: str
    ) -> tuple[str, float, bool, bool, dict[str, Any]]:
        if self._shell is None or self._sandbox is None:
            raise RuntimeError("EnterpriseHPCEnv step called before reset")

        command = action if isinstance(action, str) else str(action)
        self._step_count += 1

        self._shell.sendline(command)
        output = self._await_prompt(self._step_timeout)

        grade = self._scenario.grade(self._sandbox.state_root or Path("."))
        health_delta = grade.health - self._prev_health
        self._prev_health = grade.health
        reward = health_delta
        self._last_reward = reward
        terminated = grade.done
        truncated = not terminated and self._step_count >= self._max_steps

        http_code = self._probe_ood_code()
        info: dict[str, Any] = {
            "task_id": self._scenario.TASK_ID,
            "step": self._step_count,
            "max_steps": self._max_steps,
            "reward_source": "grader",
            "command": command,
            "grader_health": grade.health,
            "grader_details": grade.details,
            "ood_http_code": http_code,
        }
        return output, reward, terminated, truncated, info

    def render(self) -> None:
        return None

    def close(self) -> None:
        self._close_shell()
        if self._sandbox is not None:
            try:
                self._sandbox.destroy()
            except Exception as exc:
                print(f"hpc_gym sandbox destroy failed {type(exc).__name__.lower()} {exc}")
            self._sandbox = None
            self._sandbox_scenario_id = None
        if self._tmp_task_dir is not None:
            shutil.rmtree(self._tmp_task_dir, ignore_errors=True)
            self._tmp_task_dir = None

    def __enter__(self) -> "EnterpriseHPCEnv":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.close()
        return False

    def _select_scenario(self, options: dict[str, Any] | None) -> None:
        if options and "scenario" in options:
            self._scenario = resolve_scenario(options["scenario"])
            return
        if len(self._scenario_pool) == 1:
            self._scenario = self._scenario_pool[0]
            return
        self._scenario = self._rng.choice(self._scenario_pool)

    def _ensure_task_root(self) -> Path:
        if self._configured_task_root is not None:
            root = Path(self._configured_task_root)
            if self._tmp_task_dir is None:
                root.mkdir(parents=True, exist_ok=True)
        else:
            if self._tmp_task_dir is not None:
                shutil.rmtree(self._tmp_task_dir, ignore_errors=True)
            self._tmp_task_dir = tempfile.mkdtemp(prefix="hpc_task_")
            root = Path(self._tmp_task_dir)
        self._scenario.prepare_filesystem(root)
        return root

    def _spawn_shell(self) -> None:
        if self._sandbox is None:
            raise RuntimeError("sandbox must be created before shell spawn")

        bwrap_cmd = self._sandbox._build_bwrap_command(
            "exec /bin/bash --noprofile --norc -i"
        )
        env = {
            "PATH": "/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin",
            "HOME": "/root",
            "TERM": "xterm",
            "HOSTNAME": PRIMARY_HOSTNAME,
            "PS1": f"[root@{PRIMARY_HOSTNAME} \\W]\\$ ",
            "LANG": "C.UTF-8",
        }
        print(f"hpc_gym spawning pexpect bwrap {bwrap_cmd[0]}")
        self._shell = pexpect.spawn(
            bwrap_cmd[0],
            bwrap_cmd[1:],
            timeout=self._shell_timeout,
            encoding="utf-8",
            codec_errors="replace",
            env=env,
        )
        self._shell.setecho(False)

    def _bootstrap_primary_prompt(self) -> None:
        if self._shell is None:
            raise RuntimeError("shell not spawned")
        # disable bracketed-paste mode (\e[?2004l) so terminal escape sequences
        # do not pollute command output and confuse the prompt regex
        self._shell.sendline(
            "printf '\\e[?2004l'; "
            f"export PS1='[root@{PRIMARY_HOSTNAME} \\W]\\$ '; "
            "export PROMPT_COMMAND=''; stty -echo 2>/dev/null; true"
        )
        self._await_prompt(self._shell_timeout)

    @staticmethod
    def _find_python3_in_sandbox() -> str:
        """return the first python3 binary that exists on the host and will
        therefore be available inside the bwrap ro-bind at the same path."""
        candidates = [
            "/usr/bin/python3",
            "/usr/bin/python3.11",
            "/usr/bin/python3.12",
            "/usr/bin/python3.9",
            "/usr/bin/python",
        ]
        for c in candidates:
            if Path(c).exists():
                return c
        return "python3"  # fallback, may fail — caught by grace window

    def _launch_ood_daemon(self) -> None:
        if self._shell is None or self._sandbox is None:
            raise RuntimeError("shell or sandbox missing for ood launch")

        python3 = self._find_python3_in_sandbox()
        self._shell.sendline(
            f"nohup {python3} {OOD_DAEMON_SCRIPT} >{OOD_LOG_PATH} 2>&1 & disown; true"
        )
        self._await_prompt(self._shell_timeout)

        for attempt in range(20):
            code = self._probe_ood_code()
            if code in {"200", "502"}:
                self._ood_started = True
                print(f"hpc_gym ood ready http_code {code} attempts {attempt + 1}")
                return
            time.sleep(0.1)
        print("hpc_gym ood did not respond within grace window proceeding anyway")

    def _enter_login_node(self) -> None:
        if self._shell is None:
            raise RuntimeError("shell not spawned")
        self._shell.sendline("ssh login")
        self._await_prompt(self._shell_timeout)

    def _await_prompt(self, timeout: float) -> str:
        if self._shell is None:
            raise RuntimeError("shell not spawned")
        try:
            self._shell.expect(PROMPT_PATTERN, timeout=timeout)
            before = self._shell.before or ""
        except pexpect.exceptions.TIMEOUT:
            before = self._shell.before or ""
            print("hpc_gym prompt timeout sending ctrl-c to recover")
            try:
                self._shell.sendcontrol("c")
                self._shell.expect(PROMPT_PATTERN, timeout=5)
            except Exception as exc:
                print(f"hpc_gym recovery failed {type(exc).__name__.lower()} {exc}")
        except pexpect.exceptions.EOF:
            before = self._shell.before or ""
            print("hpc_gym shell eof observed")
        return _strip_ansi(before).lstrip("\r\n")

    def _probe_ood_code(self) -> str:
        if self._sandbox is None:
            return ""
        probe = self._sandbox.execute(
            f"curl -s -o /dev/null -w '%{{http_code}}' http://127.0.0.1:{OOD_PORT}/",
            timeout=10.0,
        )
        return (probe.stdout or "").strip()

    def _close_shell(self) -> None:
        if self._shell is None:
            return
        try:
            if self._shell.isalive():
                self._shell.sendline("exit 0")
                try:
                    self._shell.expect(pexpect.exceptions.EOF, timeout=2)
                except Exception:
                    pass
            self._shell.close(force=True)
        except Exception as exc:
            print(f"hpc_gym shell close failed {type(exc).__name__.lower()} {exc}")
        self._shell = None


_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


def register_env() -> None:
    try:
        gym.register(
            id="EnterpriseHPC-v0",
            entry_point="hpc_gym:EnterpriseHPCEnv",
            max_episode_steps=hpc_outage.build_definition("").metadata.max_steps,
        )
    except gym.error.Error as exc:
        print(f"hpc_gym register skipped {type(exc).__name__.lower()} {exc}")


def main() -> None:
    env = EnterpriseHPCEnv(scenario_pool=list(SCENARIO_REGISTRY))
    try:
        obs, info = env.reset(seed=0)
        print(f"reset observation {obs[:120]}")
        print(f"reset info {info}")
        obs, reward, terminated, truncated, info = env.step("sinfo")
        print(f"step reward {reward} terminated {terminated} truncated {truncated}")
        print(f"step info {info}")
        print(f"step observation\n{obs}")
    finally:
        env.close()


if __name__ == "__main__":
    os.environ.setdefault("OOD_PORT", str(OOD_PORT))
    main()
