from __future__ import annotations

import json
import re
from pathlib import Path

from sysadmin_env.models import DiagnosticTrigger
from sysadmin_env.models import DifficultyTier
from sysadmin_env.models import TaskMetadata
from sysadmin_env.models import TaskScenarioDefinition
from sysadmin_env.models import TaskScenarioState
from sysadmin_env.tasks import hpc_outage


TASK_ID = "hpc_pid_stale"
COMPLETION_HEALTH = 1.0

SHARED_STATE_PATH = hpc_outage.SHARED_STATE_PATH
COMPUTE_ROOT = hpc_outage.COMPUTE_ROOT
STALE_PID_RELATIVE = Path("var/run/slurmd.pid")
STALE_PID_PATH = COMPUTE_ROOT / STALE_PID_RELATIVE
STALE_PID_CONTENTS = "31337\n"

INITIAL_STATE: dict = {
    "cluster": "rocky-hpc",
    "cores_total": hpc_outage.CLUSTER_CORES_TOTAL,
    "cores_per_node": hpc_outage.CLUSTER_CORES_PER_NODE,
    "partitions": {
        "compute": {"nodes": ["compute-01"], "default": True},
    },
    "nodes": {
        "login": {
            "state": "up",
            "reason": "",
            "cores": hpc_outage.CLUSTER_CORES_PER_NODE,
        },
        "compute-01": {
            "state": "drain",
            "reason": "slurmd failed stale pid file at /var/run/slurmd.pid",
            "cores": hpc_outage.CLUSTER_CORES_PER_NODE,
        },
    },
    "services": {
        "slurmd@login": "active",
        "slurmd@compute-01": "failed",
        "slurmctld@login": "active",
    },
    "jobs": [
        {
            "id": 9117,
            "name": "lattice_qcd",
            "user": "physicist",
            "state": "PD",
            "partition": "compute",
            "nodes": "(NodeDown)",
            "time": "0:00",
        },
    ],
}


def build_definition(base_filesystem_path: str) -> TaskScenarioDefinition:
    metadata = TaskMetadata(
        task_id=TASK_ID,
        difficulty=DifficultyTier.hard,
        description="slurmd refuses to restart after reboot because a stale pid file is still on disk",
        max_steps=90,
        time_limit=600.0,
        base_filesystem_path=base_filesystem_path,
    )
    return TaskScenarioDefinition(
        metadata=metadata,
        requires_network_isolation=False,
        allows_nested_sandbox=True,
        diagnostic_triggers=diagnostic_triggers(),
    )


def diagnostic_triggers() -> list[DiagnosticTrigger]:
    return [
        DiagnosticTrigger(
            fact_id="cluster_queue_inspected",
            command_patterns=[r"\bsinfo\b", r"\bsqueue\b"],
            reward=0.06,
        ),
        DiagnosticTrigger(
            fact_id="compute_node_entered",
            command_patterns=[r"\bssh\s+compute-01\b"],
            reward=0.07,
        ),
        DiagnosticTrigger(
            fact_id="slurmd_service_checked",
            command_patterns=[r"systemctl\s+status\s+slurmd", r"systemctl\s+is-failed\s+slurmd"],
            reward=0.05,
        ),
        DiagnosticTrigger(
            fact_id="pid_file_inspected",
            command_patterns=[r"cat\s+.+slurmd\.pid", r"ls\s+.+run/slurmd\.pid", r"stat\s+.+slurmd\.pid"],
            reward=0.05,
        ),
        DiagnosticTrigger(
            fact_id="ood_portal_probed",
            command_patterns=[r"curl\s+.+localhost:8080", r"curl\s+.+127\.0\.0\.1:8080"],
            reward=0.05,
        ),
    ]


def prepare_filesystem(root: str | Path) -> None:
    root_path = Path(root)
    hpc_outage.prepare_filesystem(root_path)

    route_path = root_path / hpc_outage.COMPUTE_ROUTE_PATH
    route_path.parent.mkdir(parents=True, exist_ok=True)
    route_path.write_text(hpc_outage.FIXED_ROUTE)

    pid_path = root_path / STALE_PID_PATH
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.write_text(STALE_PID_CONTENTS)

    _write_state(root_path / SHARED_STATE_PATH, INITIAL_STATE)


def inject_fault(root: str | Path) -> None:
    prepare_filesystem(root)


def observe_command(root: str | Path, command: str, _result) -> None:
    _ = Path(root)
    _ = command


def synchronize(root: str | Path) -> None:
    root_path = Path(root)
    if not (root_path / SHARED_STATE_PATH).exists():
        _write_state(root_path / SHARED_STATE_PATH, INITIAL_STATE)


def grade(root: str | Path) -> TaskScenarioState:
    root_path = Path(root)
    pid_path = root_path / STALE_PID_PATH
    state_doc = _read_state(root_path / SHARED_STATE_PATH)

    pid_removed = not pid_path.exists()
    slurmd_service = state_doc.get("services", {}).get("slurmd@compute-01", "")
    slurmd_active = slurmd_service == "active"
    node_state = state_doc.get("nodes", {}).get("compute-01", {}).get("state", "")
    node_idle = node_state == "idle"

    health = 0.0
    if pid_removed:
        health += 0.3
    if slurmd_active:
        health += 0.3
    if pid_removed and slurmd_active and node_idle:
        health = COMPLETION_HEALTH

    done = pid_removed and slurmd_active and node_idle

    return TaskScenarioState(
        health=health,
        done=done,
        details={
            "stale_pid_file_removed": pid_removed,
            "slurmd_service_active": slurmd_active,
            "compute_node_idle": node_idle,
            "expected_pid_path": str(STALE_PID_RELATIVE),
        },
    )


def command_reveals_fact(command: str, trigger: DiagnosticTrigger) -> bool:
    return any(re.search(pattern, command, flags=re.IGNORECASE) for pattern in trigger.command_patterns)


def _write_state(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")


def _read_state(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text() or "{}")
    except json.JSONDecodeError:
        return {}
