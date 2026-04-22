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


TASK_ID = "hpc_gpu_ecc"
COMPLETION_HEALTH = 1.0

SHARED_STATE_PATH = hpc_outage.SHARED_STATE_PATH
NODES_ROOT = hpc_outage.NODES_ROOT
COMPUTE_ROOT = hpc_outage.COMPUTE_ROOT
ECC_RESET_RELATIVE = Path("var/lib/nvidia/ecc_reset.flag")
ECC_RESET_PATH = COMPUTE_ROOT / ECC_RESET_RELATIVE
NVIDIA_SMI_RELATIVE = Path("usr/local/bin/nvidia-smi")

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
            "reason": "gpu-0 uncorrectable ecc errors",
            "cores": hpc_outage.CLUSTER_CORES_PER_NODE,
        },
    },
    "services": {
        "slurmd@login": "active",
        "slurmd@compute-01": "failed",
        "slurmctld@login": "active",
        "nvidia-persistenced@compute-01": "active",
    },
    "gpus": {
        "compute-01:gpu-0": {
            "model": "NVIDIA H100 80GB HBM3",
            "state": "ecc_error",
            "ecc_vol_total": 47,
            "ecc_agg_total": 213,
        },
    },
    "jobs": [
        {
            "id": 11301,
            "name": "protein_fold",
            "user": "biogrid",
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
        description="compute node drained because nvidia-smi reports gpu-0 uncorrectable ecc errors",
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
            fact_id="gpu_status_inspected",
            command_patterns=[r"\bnvidia-smi\b(?!\s+-r)"],
            reward=0.06,
        ),
        DiagnosticTrigger(
            fact_id="ecc_counters_queried",
            command_patterns=[r"nvidia-smi\s+(-q|--query).*ecc", r"nvidia-smi\s+.*ecc"],
            reward=0.05,
        ),
        DiagnosticTrigger(
            fact_id="slurmd_service_checked",
            command_patterns=[r"systemctl\s+status\s+slurmd", r"systemctl\s+is-failed\s+slurmd"],
            reward=0.05,
        ),
    ]


def prepare_filesystem(root: str | Path) -> None:
    root_path = Path(root)
    hpc_outage.prepare_filesystem(root_path)

    route_path = root_path / hpc_outage.COMPUTE_ROUTE_PATH
    route_path.parent.mkdir(parents=True, exist_ok=True)
    route_path.write_text(hpc_outage.FIXED_ROUTE)

    ecc_path = root_path / ECC_RESET_PATH
    ecc_path.parent.mkdir(parents=True, exist_ok=True)
    if ecc_path.exists():
        ecc_path.unlink()

    _write_state(root_path / SHARED_STATE_PATH, INITIAL_STATE)

    _write_executable(root_path / NVIDIA_SMI_RELATIVE, _login_nvidia_smi_stub())
    compute_bin = root_path / COMPUTE_ROOT / "usr/local/bin"
    compute_bin.mkdir(parents=True, exist_ok=True)
    _write_executable(compute_bin / "nvidia-smi", _compute_nvidia_smi_stub())


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

    ecc_reset = (root_path / ECC_RESET_PATH).exists()
    gpu_state = (
        state_doc.get("gpus", {})
        .get("compute-01:gpu-0", {})
        .get("state", "")
    )
    gpu_healthy = gpu_state == "healthy"

    slurmd_service = state_doc.get("services", {}).get("slurmd@compute-01", "")
    slurmd_active = slurmd_service == "active"
    node_state = state_doc.get("nodes", {}).get("compute-01", {}).get("state", "")
    node_idle = node_state == "idle"

    health = 0.0
    if ecc_reset:
        health += 0.25
    if gpu_healthy:
        health += 0.25
    if slurmd_active:
        health += 0.2
    if ecc_reset and gpu_healthy and slurmd_active and node_idle:
        health = COMPLETION_HEALTH

    done = ecc_reset and gpu_healthy and slurmd_active and node_idle

    return TaskScenarioState(
        health=health,
        done=done,
        details={
            "ecc_reset_sentinel_present": ecc_reset,
            "gpu_healthy": gpu_healthy,
            "slurmd_service_active": slurmd_active,
            "compute_node_idle": node_idle,
            "gpu_state": gpu_state or "unknown",
            "expected_sentinel_path": str(ECC_RESET_RELATIVE),
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


def _login_nvidia_smi_stub() -> str:
    # on the login node there is no gpu the agent must ssh into compute-01
    return """#!/bin/sh
echo "nvidia-smi: no devices were found" >&2
exit 9
"""


def _compute_nvidia_smi_stub() -> str:
    return """#!/usr/bin/env python3
import argparse
import fcntl
import json
import os
import sys

STATE_PATH = "/mnt/shared/slurm_state.json"
ECC_SENTINEL = "/var/lib/nvidia/ecc_reset.flag"
GPU_KEY = "compute-01:gpu-0"

def read_state():
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as fh:
            fcntl.flock(fh.fileno(), fcntl.LOCK_SH)
            try:
                raw = fh.read()
            finally:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        return json.loads(raw or "{}")
    except FileNotFoundError:
        return {}

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

def render_query(doc):
    gpu = doc.get("gpus", {}).get(GPU_KEY, {})
    model = gpu.get("model", "unknown")
    state = gpu.get("state", "unknown")
    vol = gpu.get("ecc_vol_total", 0)
    agg = gpu.get("ecc_agg_total", 0)
    print(f"==============NVSMI LOG==============")
    print(f"GPU 00000000:17:00.0  {model}")
    print(f"    Product State         : {state}")
    print(f"    ECC Errors")
    print(f"        Volatile")
    print(f"            Total         : {vol}")
    print(f"        Aggregate")
    print(f"            Total         : {agg}")

def render_summary(doc):
    gpu = doc.get("gpus", {}).get(GPU_KEY, {})
    state = gpu.get("state", "unknown")
    note = "ECC" if state != "healthy" else "OK"
    print(f"+-----------------------------------------------------------------------------+")
    print(f"| NVIDIA-SMI 555.42.02   Driver Version: 555.42.02   CUDA Version: 12.5       |")
    print(f"|-----------------------------------------------------------------------------|")
    print(f"| GPU  Name                    Bus-Id          Pwr:Usage/Cap |  Memory  {note:<4} |")
    print(f"|  0   {gpu.get('model','unknown'):<24} 0000:17:00.0     78W / 700W |    0MiB {note:<5} |")
    print(f"+-----------------------------------------------------------------------------+")

def handle_reset(gpu_id):
    open(ECC_SENTINEL, "w").close()
    def apply(doc):
        gpus = doc.setdefault("gpus", {})
        entry = gpus.setdefault(GPU_KEY, {})
        entry["state"] = "healthy"
        entry["ecc_vol_total"] = 0
        services = doc.setdefault("services", {})
        services["slurmd@compute-01"] = "active"
        nodes = doc.setdefault("nodes", {})
        compute = nodes.setdefault("compute-01", {})
        compute["state"] = "idle"
        compute["reason"] = ""
    mutate_state(apply)
    print(f"GPU {gpu_id}: ECC error counters reset. Node returned to idle.")
    return 0

def main(argv):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-r", "--reset", action="store_true")
    parser.add_argument("-i", "--id", default="0")
    parser.add_argument("-q", "--query", action="store_true")
    parser.add_argument("-d", "--display", default="")
    parser.add_argument("--help", action="store_true")
    try:
        args, extra = parser.parse_known_args(argv[1:])
    except SystemExit:
        return 2
    if args.help:
        print("nvidia-smi [-q] [-d ECC] [-r -i <gpu>]")
        return 0
    os.makedirs(os.path.dirname(ECC_SENTINEL), exist_ok=True)
    doc = read_state()
    if args.reset:
        return handle_reset(args.id)
    if args.query:
        render_query(doc)
        return 0
    render_summary(doc)
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv))
"""
