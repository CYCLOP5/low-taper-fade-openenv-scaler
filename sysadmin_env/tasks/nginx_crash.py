from __future__ import annotations

import re
from pathlib import Path

from sysadmin_env.models import DiagnosticTrigger
from sysadmin_env.models import DifficultyTier
from sysadmin_env.models import TaskMetadata
from sysadmin_env.models import TaskScenarioDefinition
from sysadmin_env.models import TaskScenarioState


TASK_ID = "nginx_crash"
PID_PATH = Path("var/run/nginx.pid")
CONFIG_PATH = Path("etc/nginx/nginx.conf")
ERROR_LOG_PATH = Path("var/log/nginx/error.log")
RUNNING_FLAG_PATH = Path("run/nginx.running")
PORT_FLAG_PATH = Path("run/nginx.port")

BROKEN_CONFIG = """worker_processes 1;
events {
    worker_connections 128;
}
http {
    server {
        listen 8080
        location / {
            return 200 ok;
        }
    }
}
"""

FIXED_CONFIG = """worker_processes 1;
events {
    worker_connections 128;
}
http {
    server {
        listen 8080;
        location / {
            return 200 ok;
        }
    }
}
"""

ERROR_LOG = "nginx emerg unexpected end of statement in /etc/nginx/nginx.conf line 7\n"


def build_definition(base_filesystem_path: str) -> TaskScenarioDefinition:
    metadata = TaskMetadata(
        task_id=TASK_ID,
        difficulty=DifficultyTier.easy,
        description="nginx crashed with stale pid and config syntax error",
        max_steps=40,
        time_limit=300.0,
        base_filesystem_path=base_filesystem_path,
    )
    return TaskScenarioDefinition(
        metadata=metadata,
        requires_network_isolation=False,
        diagnostic_triggers=diagnostic_triggers(),
    )


def diagnostic_triggers() -> list[DiagnosticTrigger]:
    return [
        DiagnosticTrigger(
            fact_id="nginx_error_log_checked",
            command_patterns=[r"cat\s+.+error\.log", r"tail\s+.+error\.log", r"grep\s+.+error\.log"],
            reward=0.05,
        ),
        DiagnosticTrigger(
            fact_id="nginx_config_tested",
            command_patterns=[r"nginx\s+-t", r"/usr/sbin/nginx\s+-t"],
            reward=0.08,
        ),
        DiagnosticTrigger(
            fact_id="nginx_pid_checked",
            command_patterns=[r"cat\s+.+nginx\.pid", r"sed\s+.+nginx\.pid"],
            reward=0.04,
        ),
        DiagnosticTrigger(
            fact_id="nginx_process_table_checked",
            command_patterns=[r"ps\b.*nginx", r"pgrep\b.*nginx"],
            reward=0.04,
        ),
    ]


def prepare_filesystem(root: str | Path) -> None:
    root_path = Path(root)
    for relative in [
        Path("etc/nginx"),
        Path("var/run"),
        Path("var/log/nginx"),
        Path("run"),
        Path("usr/local/bin"),
        Path("root"),
        Path("tmp"),
        Path("home"),
    ]:
        (root_path / relative).mkdir(parents=True, exist_ok=True)

    (root_path / CONFIG_PATH).write_text(BROKEN_CONFIG)
    (root_path / PID_PATH).write_text("424242\n")
    (root_path / ERROR_LOG_PATH).write_text(ERROR_LOG)
    (root_path / RUNNING_FLAG_PATH).write_text("stopped\n")
    (root_path / PORT_FLAG_PATH).write_text("8080\n")
    _write_executable(root_path / "usr/local/bin/nginx", _nginx_stub())
    _write_executable(root_path / "usr/local/bin/curl", _curl_stub())
    _write_executable(root_path / "usr/local/bin/ps", _ps_stub())
    _write_executable(root_path / "usr/local/bin/pgrep", _pgrep_stub())
    _write_executable(root_path / "usr/local/bin/service", _service_stub())
    _write_executable(root_path / "usr/local/bin/systemctl", _systemctl_stub())


def inject_fault(root: str | Path) -> None:
    prepare_filesystem(root)


def observe_command(root: str | Path, command: str, _result) -> None:
    root_path = Path(root)
    if re.search(r"\b(service|systemctl)\b.*\bstatus\b", command, flags=re.IGNORECASE):
        if _service_is_running(root_path):
            (root_path / RUNNING_FLAG_PATH).write_text("running\n")
        else:
            (root_path / RUNNING_FLAG_PATH).write_text("stopped\n")


def synchronize(root: str | Path) -> None:
    root_path = Path(root)
    running_flag = root_path / RUNNING_FLAG_PATH
    if not running_flag.exists():
        running_flag.write_text("stopped\n")


def grade(root: str | Path) -> TaskScenarioState:
    root_path = Path(root)
    stale_pid_removed = _stale_pid_cleared(root_path / PID_PATH)
    config_fixed = _config_is_fixed(root_path / CONFIG_PATH)
    running = _service_is_running(root_path)

    health = 0.0
    if stale_pid_removed:
        health += 0.25
    if config_fixed:
        health += 0.35
    if running:
        health += 0.4

    return TaskScenarioState(
        health=health,
        done=running,
        details={
            "stale_pid_removed": stale_pid_removed,
            "config_fixed": config_fixed,
            "service_running": running,
        },
    )


def command_reveals_fact(command: str, trigger: DiagnosticTrigger) -> bool:
    return any(re.search(pattern, command, flags=re.IGNORECASE) for pattern in trigger.command_patterns)


def _config_is_fixed(config_path: Path) -> bool:
    if not config_path.exists():
        return False
    config_text = config_path.read_text()
    return re.search(r"listen\s+8080\s*;", config_text) is not None


def _stale_pid_cleared(pid_path: Path) -> bool:
    if not pid_path.exists():
        return True
    return pid_path.read_text().strip() == "1234"


def _service_is_running(root_path: Path) -> bool:
    if not _config_is_fixed(root_path / CONFIG_PATH):
        return False
    running_flag = root_path / RUNNING_FLAG_PATH
    return running_flag.exists() and running_flag.read_text().strip() == "running"


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content)
    path.chmod(0o755)


def _nginx_stub() -> str:
    return """#!/bin/sh
config="/etc/nginx/nginx.conf"
pidfile="/var/run/nginx.pid"
statefile="/run/nginx.running"
if [ "$1" = "-t" ]; then
    if grep -Eq 'listen[[:space:]]+8080;' "$config"; then
        echo "nginx config ok"
        exit 0
    fi
    printf '%s\n' "nginx emerg unexpected end of statement in /etc/nginx/nginx.conf line 7" >&2
    exit 1
fi
if [ "$1" = "-s" ] && [ "$2" = "stop" ]; then
    rm -f "$pidfile"
    printf '%s\n' stopped > "$statefile"
    exit 0
fi
if [ -f "$pidfile" ] && [ "$(cat "$pidfile" 2>/dev/null)" != "1234" ]; then
    printf '%s\n' "nginx stale pid present" >&2
    exit 1
fi
if grep -Eq 'listen[[:space:]]+8080;' "$config"; then
    printf '%s\n' running > "$statefile"
    printf '%s\n' 1234 > "$pidfile"
    printf '%s\n' "nginx started"
    exit 0
fi
printf '%s\n' "nginx failed to start" >&2
exit 1
"""


def _curl_stub() -> str:
    return """#!/bin/sh
state="$(cat /run/nginx.running 2>/dev/null || printf '%s' stopped)"
if [ "$state" = "running" ]; then
    printf '%s\n' "HTTP/1.1 200 ok"
    printf '%s\n' "server: nginx"
    exit 0
fi
printf '%s\n' "curl failed to connect" >&2
exit 7
"""


def _ps_stub() -> str:
    return """#!/bin/sh
printf '%s\n' "USER PID CMD"
state="$(cat /run/nginx.running 2>/dev/null || printf '%s' stopped)"
if [ "$state" = "running" ]; then
    printf '%s\n' "root 1234 nginx"
fi
exit 0
"""


def _pgrep_stub() -> str:
    return """#!/bin/sh
state="$(cat /run/nginx.running 2>/dev/null || printf '%s' stopped)"
if [ "$state" = "running" ]; then
    printf '%s\n' 1234
    exit 0
fi
exit 1
"""


def _service_stub() -> str:
    return """#!/bin/sh
name="$1"
action="$2"
if [ "$name" = "nginx" ] && [ "$action" = "start" ]; then
    exec nginx
fi
if [ "$name" = "nginx" ] && [ "$action" = "stop" ]; then
    exec nginx -s stop
fi
if [ "$name" = "nginx" ] && [ "$action" = "status" ]; then
    state="$(cat /run/nginx.running 2>/dev/null || printf '%s' stopped)"
    if [ "$state" = "running" ]; then
        printf '%s\n' "nginx active"
        exit 0
    fi
    printf '%s\n' "nginx inactive" >&2
    exit 3
fi
printf '%s\n' "unsupported service action" >&2
exit 1
"""


def _systemctl_stub() -> str:
    return """#!/bin/sh
action="$1"
name="$2"
if [ "$name" = "nginx" ] && [ "$action" = "start" ]; then
    exec nginx
fi
if [ "$name" = "nginx" ] && [ "$action" = "stop" ]; then
    exec nginx -s stop
fi
if [ "$name" = "nginx" ] && [ "$action" = "restart" ]; then
    nginx -s stop >/dev/null 2>&1 || true
    exec nginx
fi
if [ "$name" = "nginx" ] && [ "$action" = "status" ]; then
    exec service nginx status
fi
printf '%s\n' "unsupported systemctl action" >&2
exit 1
"""
