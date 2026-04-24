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


TASK_ID = "hpc_nfs_stale"
COMPLETION_HEALTH = 1.0

SHARED_STATE_PATH = hpc_outage.SHARED_STATE_PATH
NODES_ROOT = hpc_outage.NODES_ROOT
COMPUTE_ROOT = hpc_outage.COMPUTE_ROOT
MOUNT_STALE_RELATIVE = Path("mnt/shared/.mount_stale")
MOUNT_VALID_RELATIVE = Path("mnt/shared/.mount_valid")
MOUNT_STUB_RELATIVE = Path("usr/local/bin/mount")
UMOUNT_STUB_RELATIVE = Path("usr/local/bin/umount")

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
            "reason": "nfs /mnt/shared stale file handle",
            "cores": hpc_outage.CLUSTER_CORES_PER_NODE,
        },
    },
    "services": {
        "slurmd@login": "active",
        "slurmd@compute-01": "failed",
        "slurmctld@login": "active",
        "rpc-statd@compute-01": "active",
    },
    "jobs": [
        {
            "id": 10244,
            "name": "cryoem_refine",
            "user": "structures",
            "state": "PD",
            "partition": "compute",
            "nodes": "(NfsStale)",
            "time": "0:00",
        },
    ],
}


def build_definition(base_filesystem_path: str) -> TaskScenarioDefinition:
    metadata = TaskMetadata(
        task_id=TASK_ID,
        difficulty=DifficultyTier.hard,
        description="compute node drained because the nfs share at /mnt/shared reports stale file handle",
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
            fact_id="share_inspected",
            command_patterns=[r"ls\s+.+/mnt/shared", r"stat\s+.+/mnt/shared", r"\bmount\b"],
            reward=0.05,
        ),
        DiagnosticTrigger(
            fact_id="slurmd_service_checked",
            command_patterns=[r"systemctl\s+status\s+slurmd", r"systemctl\s+is-failed\s+slurmd"],
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

    valid_path = root_path / MOUNT_VALID_RELATIVE
    if valid_path.exists():
        valid_path.unlink()

    stale_path = root_path / MOUNT_STALE_RELATIVE
    stale_path.parent.mkdir(parents=True, exist_ok=True)
    stale_path.write_text("stale nfs file handle detected at mount time\n")

    _write_executable(root_path / MOUNT_STUB_RELATIVE, _mount_stub())
    _write_executable(root_path / UMOUNT_STUB_RELATIVE, _umount_stub())
    compute_bin = root_path / COMPUTE_ROOT / "usr/local/bin"
    compute_bin.mkdir(parents=True, exist_ok=True)
    _write_executable(compute_bin / "mount", _mount_stub())
    _write_executable(compute_bin / "umount", _umount_stub())
    _write_executable(compute_bin / "systemctl", _systemctl_nfs_stub())
    _write_executable(root_path / "usr/local/bin/systemctl", _systemctl_nfs_stub())

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
    state_doc = _read_state(root_path / SHARED_STATE_PATH)

    stale_gone = not (root_path / MOUNT_STALE_RELATIVE).exists()
    mount_valid = (root_path / MOUNT_VALID_RELATIVE).exists()
    slurmd_service = state_doc.get("services", {}).get("slurmd@compute-01", "")
    slurmd_active = slurmd_service == "active"
    node_state = state_doc.get("nodes", {}).get("compute-01", {}).get("state", "")
    node_idle = node_state == "idle"

    health = 0.0
    if stale_gone:
        health += 0.2
    if mount_valid:
        health += 0.3
    if slurmd_active:
        health += 0.2
    if stale_gone and mount_valid and slurmd_active and node_idle:
        health = COMPLETION_HEALTH

    done = stale_gone and mount_valid and slurmd_active and node_idle

    return TaskScenarioState(
        health=health,
        done=done,
        details={
            "stale_marker_removed": stale_gone,
            "mount_valid_sentinel_present": mount_valid,
            "slurmd_service_active": slurmd_active,
            "compute_node_idle": node_idle,
            "expected_valid_sentinel": str(MOUNT_VALID_RELATIVE),
        },
    )


def command_reveals_fact(command: str, trigger: DiagnosticTrigger) -> bool:
    return any(re.search(pattern, command, flags=re.IGNORECASE) for pattern in trigger.command_patterns)


def _write_executable(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    path.chmod(0o755)


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


def _mount_stub() -> str:
    return """#!/bin/sh
TARGET=""
for arg in "$@"; do
    case "$arg" in
        -*|-t|nfs|-o) ;;
        *) TARGET="$arg" ;;
    esac
done
if [ -z "$TARGET" ]; then
    if [ -f /mnt/shared/.mount_stale ]; then
        echo "shared.hpc.local:/srv/shared on /mnt/shared type nfs4 (rw,stale)" 
    elif [ -f /mnt/shared/.mount_valid ]; then
        echo "shared.hpc.local:/srv/shared on /mnt/shared type nfs4 (rw,fresh)"
    else
        echo "shared.hpc.local:/srv/shared on /mnt/shared type nfs4 (rw,idle)"
    fi
    exit 0
fi
case "$TARGET" in
    /mnt/shared|shared)
        if [ -f /mnt/shared/.mount_stale ]; then
            echo "mount: /mnt/shared already bound with stale handle. umount first." >&2
            exit 32
        fi
        if [ -f /mnt/shared/.mount_valid ]; then
            echo "mount: /mnt/shared already mounted" >&2
            exit 32
        fi
        mkdir -p /mnt/shared
        printf 'fresh mount handle\\n' > /mnt/shared/.mount_valid
        echo "mounting shared.hpc.local:/srv/shared on /mnt/shared type nfs4"
        exit 0
        ;;
    *)
        echo "mount: $TARGET is not a known mount target in this sandbox" >&2
        exit 32
        ;;
esac
"""


def _umount_stub() -> str:
    return """#!/bin/sh
LAZY=0
TARGET=""
for arg in "$@"; do
    case "$arg" in
        -l|--lazy) LAZY=1 ;;
        -f|--force) ;;
        -*) ;;
        *) TARGET="$arg" ;;
    esac
done
if [ -z "$TARGET" ]; then
    echo "umount: missing target" >&2
    exit 1
fi
case "$TARGET" in
    /mnt/shared|shared)
        if [ -f /mnt/shared/.mount_stale ]; then
            rm -f /mnt/shared/.mount_stale
            echo "umount: /mnt/shared detached (lazy=$LAZY)"
            exit 0
        fi
        if [ -f /mnt/shared/.mount_valid ]; then
            rm -f /mnt/shared/.mount_valid
            echo "umount: /mnt/shared detached clean (lazy=$LAZY)"
            exit 0
        fi
        echo "umount: /mnt/shared not mounted" >&2
        exit 32
        ;;
    *)
        echo "umount: $TARGET is not a known mount in this sandbox" >&2
        exit 32
        ;;
esac
"""


def _systemctl_nfs_stub() -> str:
    return """#!/usr/bin/env python3
import fcntl
import json
import os
import socket
import sys

STATE_PATH = "/mnt/shared/slurm_state.json"
MOUNT_VALID = "/mnt/shared/.mount_valid"
MOUNT_STALE = "/mnt/shared/.mount_stale"

def current_hostname():
    host = os.environ.get("HOSTNAME")
    if host:
        return host.strip()
    try:
        return socket.gethostname()
    except OSError:
        return ""

def mount_is_fresh():
    return os.path.isfile(MOUNT_VALID) and not os.path.isfile(MOUNT_STALE)

def mutate_state(mutator):
    with open(STATE_PATH, "r+", encoding="utf-8") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            raw = fh.read()
            doc = json.loads(raw or "{}")
            mutator(doc)
            fh.seek(0)
            fh.truncate()
            fh.write(json.dumps(doc, indent=2, sort_keys=True) + "\\n")
            fh.flush()
            os.fsync(fh.fileno())
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)

def read_state():
    with open(STATE_PATH, "r", encoding="utf-8") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_SH)
        try:
            raw = fh.read()
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    return json.loads(raw or "{}")

def unit_key(unit, host):
    base = unit.split(".")[0]
    if "@" in base:
        return base
    return f"{base}@{host}" if host else base

def handle_status(unit, host):
    try:
        doc = read_state()
    except FileNotFoundError:
        sys.stderr.write("systemctl: slurm state file is missing\\n")
        return 3
    key = unit_key(unit, host)
    status = doc.get("services", {}).get(key, "inactive")
    if status == "active":
        print(f"{unit} - loaded active (running)")
        return 0
    if status == "failed":
        print(f"{unit} - loaded failed (Result: exit-code)")
        return 3
    print(f"{unit} - loaded inactive (dead)")
    return 3

def handle_is_failed(unit, host):
    try:
        doc = read_state()
    except FileNotFoundError:
        print("unknown")
        return 1
    key = unit_key(unit, host)
    status = doc.get("services", {}).get(key, "inactive")
    print(status)
    return 0 if status == "failed" else 1

def handle_restart(unit, host):
    base = unit.split(".")[0].split("@")[0]
    if base != "slurmd":
        def noop(doc):
            services = doc.setdefault("services", {})
            services[unit_key(unit, host)] = "active"
        mutate_state(noop)
        print(f"{unit} restarted")
        return 0

    unit_no_svc = unit.split(".")[0]
    explicit_target = unit_no_svc.split("@", 1)[1] if "@" in unit_no_svc else None
    effective_host = explicit_target if explicit_target else host

    if effective_host != "compute-01":
        def remote_restart(doc):
            services = doc.setdefault("services", {})
            services[f"slurmd@{effective_host or 'unknown'}"] = "active"
        mutate_state(remote_restart)
        print(f"{unit} restarted on {effective_host or 'unknown'}")
        return 0

    fresh = mount_is_fresh()

    def apply(doc):
        services = doc.setdefault("services", {})
        nodes = doc.setdefault("nodes", {})
        compute = nodes.setdefault("compute-01", {})
        if fresh:
            services["slurmd@compute-01"] = "active"
            compute["state"] = "idle"
            compute["reason"] = ""
        else:
            services["slurmd@compute-01"] = "failed"
            compute["state"] = "drain"
            compute["reason"] = "nfs /mnt/shared stale file handle"

    mutate_state(apply)

    if fresh:
        print("slurmd restarted on compute-01 node returned to idle")
        return 0
    sys.stderr.write("slurmd failed to come up /mnt/shared is stale run umount and mount first\\n")
    return 1

def main(argv):
    if len(argv) < 2:
        sys.stderr.write("systemctl: missing command\\n")
        return 1
    action = argv[1]
    rest = argv[2:]
    if action in {"daemon-reload", "list-units"}:
        print("ok")
        return 0
    if not rest:
        sys.stderr.write(f"systemctl: {action} requires a unit\\n")
        return 1
    unit = rest[0]
    host = current_hostname()
    if action == "status":
        return handle_status(unit, host)
    if action == "is-failed":
        return handle_is_failed(unit, host)
    if action in {"restart", "start"}:
        return handle_restart(unit, host)
    if action == "stop":
        def stop(doc):
            services = doc.setdefault("services", {})
            services[unit_key(unit, host)] = "inactive"
        mutate_state(stop)
        print(f"{unit} stopped")
        return 0
    sys.stderr.write(f"systemctl: unsupported action {action}\\n")
    return 1

if __name__ == "__main__":
    sys.exit(main(sys.argv))
"""
