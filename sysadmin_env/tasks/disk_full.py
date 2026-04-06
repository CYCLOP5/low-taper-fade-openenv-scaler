from __future__ import annotations

import re
from pathlib import Path

from sysadmin_env.models import DiagnosticTrigger
from sysadmin_env.models import DifficultyTier
from sysadmin_env.models import TaskMetadata
from sysadmin_env.models import TaskScenarioDefinition
from sysadmin_env.models import TaskScenarioState


TASK_ID = "disk_full"
MOUNT_PATH = Path("mnt/data")
HIDDEN_LOG_PATH = Path("mnt/data/.cache/.rotated/app.trace")
CAPACITY_PATH = Path("mnt/data/.capacity")
USAGE_PATH = Path("mnt/data/.usage")
DISCOVERY_PATH = Path("mnt/data/.diagnosed")


def build_definition(base_filesystem_path: str) -> TaskScenarioDefinition:
    metadata = TaskMetadata(
        task_id=TASK_ID,
        difficulty=DifficultyTier.medium,
        description="hidden sparse log file filling loopback mount",
        max_steps=55,
        time_limit=420.0,
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
            fact_id="disk_usage_checked",
            command_patterns=[r"df\b", r"df\s+-h"],
            reward=0.06,
        ),
        DiagnosticTrigger(
            fact_id="large_files_checked",
            command_patterns=[r"du\b", r"du\s+-sh"],
            reward=0.05,
        ),
        DiagnosticTrigger(
            fact_id="hidden_files_checked",
            command_patterns=[r"find\b.*-name", r"find\b.*-type\s+f"],
            reward=0.06,
        ),
        DiagnosticTrigger(
            fact_id="open_files_checked",
            command_patterns=[r"lsof\b", r"lsof\b.*deleted"],
            reward=0.05,
        ),
    ]


def prepare_filesystem(root: str | Path) -> None:
    root_path = Path(root)
    (root_path / MOUNT_PATH / ".cache/.rotated").mkdir(parents=True, exist_ok=True)
    (root_path / "usr/local/bin").mkdir(parents=True, exist_ok=True)
    (root_path / "root").mkdir(parents=True, exist_ok=True)
    (root_path / CAPACITY_PATH).write_text("100\n")
    (root_path / DISCOVERY_PATH).write_text("unknown\n")
    (root_path / HIDDEN_LOG_PATH).write_text("x" * 100)
    _write_executable(root_path / "usr/local/bin/df", _df_stub())
    _write_executable(root_path / "usr/local/bin/du", _du_stub())
    _write_executable(root_path / "usr/local/bin/lsof", _lsof_stub())
    synchronize(root_path)


def inject_fault(root: str | Path) -> None:
    prepare_filesystem(root)


def observe_command(root: str | Path, command: str, _result) -> None:
    root_path = Path(root)
    current_state = _usage_file_value(root_path / DISCOVERY_PATH)

    if re.search(r"\bdf\b", command, flags=re.IGNORECASE):
        current_state = "full"

    if re.search(r"\b(find|du|lsof|ls)\b", command, flags=re.IGNORECASE):
        current_state = "found"

    (root_path / DISCOVERY_PATH).write_text(f"{current_state}\n")
    synchronize(root_path)


def synchronize(root: str | Path) -> None:
    root_path = Path(root)
    capacity = int((root_path / CAPACITY_PATH).read_text().strip())
    hidden_size = 0
    if (root_path / HIDDEN_LOG_PATH).exists():
        hidden_size = len((root_path / HIDDEN_LOG_PATH).read_text())
    usage = min(hidden_size, capacity)
    (root_path / USAGE_PATH).write_text(f"{usage}\n")


def grade(root: str | Path) -> TaskScenarioState:
    root_path = Path(root)
    discovery_state = _usage_file_value(root_path / DISCOVERY_PATH)
    diagnosis_recorded = discovery_state in {"full", "found"}
    hidden_file_found = not (root_path / HIDDEN_LOG_PATH).exists() or discovery_state == "found"
    capacity_free = _free_capacity(root_path) > 0

    health = 0.0
    if diagnosis_recorded:
        health += 0.3
    if hidden_file_found:
        health += 0.3
    if capacity_free:
        health += 0.4

    return TaskScenarioState(
        health=health,
        done=capacity_free,
        details={
            "filesystem_identified": diagnosis_recorded,
            "hidden_file_found": hidden_file_found,
            "filesystem_has_capacity": capacity_free,
        },
    )


def command_reveals_fact(command: str, trigger: DiagnosticTrigger) -> bool:
    return any(re.search(pattern, command, flags=re.IGNORECASE) for pattern in trigger.command_patterns)


def _usage_file_value(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text().strip()


def _free_capacity(root_path: Path) -> int:
    capacity = int((root_path / CAPACITY_PATH).read_text().strip())
    usage = int((root_path / USAGE_PATH).read_text().strip())
    return capacity - usage


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content)
    path.chmod(0o755)


def _df_stub() -> str:
    return """#!/bin/sh
capacity="$(cat /mnt/data/.capacity 2>/dev/null || printf '%s' 100)"
usage="$(cat /mnt/data/.usage 2>/dev/null || printf '%s' 0)"
avail=$((capacity - usage))
if [ "$avail" -lt 0 ]; then
    avail=0
fi
usep=0
if [ "$capacity" -gt 0 ]; then
    usep=$((usage * 100 / capacity))
fi
printf '%s\n' "filesystem size used avail use% mounted on"
printf 'loop0 %sm %sm %sm %s%% /mnt/data\n' "$capacity" "$usage" "$avail" "$usep"
"""


def _du_stub() -> str:
    return """#!/bin/sh
size=0
if [ -f /mnt/data/.cache/.rotated/app.trace ]; then
    size=$(wc -c < /mnt/data/.cache/.rotated/app.trace)
fi
printf '%s\t%s\n' "$size" "/mnt/data/.cache/.rotated/app.trace"
printf '%s\t%s\n' "$size" "/mnt/data/.cache/.rotated"
printf '%s\t%s\n' "$size" "/mnt/data"
"""


def _lsof_stub() -> str:
    return """#!/bin/sh
if [ -f /mnt/data/.cache/.rotated/app.trace ]; then
    printf '%s\n' "python 321 root 3r REG 0 0 0 /mnt/data/.cache/.rotated/app.trace"
fi
exit 0
"""
