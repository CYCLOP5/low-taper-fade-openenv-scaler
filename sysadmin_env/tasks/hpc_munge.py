from __future__ import annotations

import json
import re
import stat
from pathlib import Path

from sysadmin_env.models import DiagnosticTrigger
from sysadmin_env.models import DifficultyTier
from sysadmin_env.models import TaskMetadata
from sysadmin_env.models import TaskScenarioDefinition
from sysadmin_env.models import TaskScenarioState
from sysadmin_env.tasks import hpc_outage


TASK_ID = "hpc_munge"
COMPLETION_HEALTH = 1.0

SHARED_STATE_PATH = hpc_outage.SHARED_STATE_PATH
NODES_ROOT = hpc_outage.NODES_ROOT
COMPUTE_ROOT = hpc_outage.COMPUTE_ROOT
MUNGE_KEY_RELATIVE = Path("etc/munge/munge.key")
MUNGE_KEY_PATH = COMPUTE_ROOT / MUNGE_KEY_RELATIVE
EXPECTED_KEY_MODE = 0o400
EXPECTED_KEY_BYTES = b"MUNGE_KEY_" + b"A" * 54 + b"\n"

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
            "reason": "munge authentication failed",
            "cores": hpc_outage.CLUSTER_CORES_PER_NODE,
        },
    },
    "services": {
        "slurmd@login": "active",
        "slurmd@compute-01": "failed",
        "slurmctld@login": "active",
        "munge@compute-01": "failed",
        "munge@login": "active",
    },
    "jobs": [
        {
            "id": 8421,
            "name": "cfd_simulation",
            "user": "engineer",
            "state": "PD",
            "partition": "compute",
            "nodes": "(AuthFail)",
            "time": "0:00",
        },
    ],
}


def build_definition(base_filesystem_path: str) -> TaskScenarioDefinition:
    metadata = TaskMetadata(
        task_id=TASK_ID,
        difficulty=DifficultyTier.hard,
        description="slurm compute node draining due to munge key permission fault and broken route",
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
            fact_id="munge_key_inspected",
            command_patterns=[r"ls\s+-l\s+.+munge", r"stat\s+.+munge\.key", r"cat\s+.+munge\.key"],
            reward=0.05,
        ),
        DiagnosticTrigger(
            fact_id="munge_service_checked",
            command_patterns=[r"systemctl\s+status\s+munge", r"systemctl\s+is-failed\s+munge"],
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

    _write_state(root_path / SHARED_STATE_PATH, INITIAL_STATE)

    (root_path / COMPUTE_ROOT / "etc/munge").mkdir(parents=True, exist_ok=True)
    key_path = root_path / MUNGE_KEY_PATH
    key_path.write_bytes(EXPECTED_KEY_BYTES)
    key_path.chmod(0o644)


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
    key_path = root_path / MUNGE_KEY_PATH

    key_locked_down = _key_mode_matches(key_path)
    key_contents_intact = _key_contents_match(key_path)
    munge_key_fixed = key_locked_down and key_contents_intact

    state_doc = _read_state(root_path / SHARED_STATE_PATH)
    node_state = (
        state_doc.get("nodes", {})
        .get("compute-01", {})
        .get("state", "")
    )
    munge_service = (
        state_doc.get("services", {}).get("munge@compute-01", "")
    )
    slurmd_service = (
        state_doc.get("services", {}).get("slurmd@compute-01", "")
    )

    auth_restored = munge_service == "active"
    node_idle = node_state == "idle" and slurmd_service == "active"

    health = 0.0
    if munge_key_fixed:
        health += 0.3
    if auth_restored:
        health += 0.3
    if node_idle:
        health = COMPLETION_HEALTH

    done = munge_key_fixed and auth_restored and node_idle

    return TaskScenarioState(
        health=health,
        done=done,
        details={
            "munge_key_mode_correct": key_locked_down,
            "munge_key_contents_correct": key_contents_intact,
            "munge_service_active": auth_restored,
            "compute_node_idle": node_idle,
            "expected_mode_octal": oct(EXPECTED_KEY_MODE),
        },
    )


def command_reveals_fact(command: str, trigger: DiagnosticTrigger) -> bool:
    return any(re.search(pattern, command, flags=re.IGNORECASE) for pattern in trigger.command_patterns)


def _key_mode_matches(path: Path) -> bool:
    if not path.exists():
        return False
    mode = stat.S_IMODE(path.stat().st_mode)
    return mode == EXPECTED_KEY_MODE


def _key_contents_match(path: Path) -> bool:
    if not path.exists():
        return False
    return path.read_bytes() == EXPECTED_KEY_BYTES


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
