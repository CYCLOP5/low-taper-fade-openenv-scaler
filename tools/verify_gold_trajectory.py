from __future__ import annotations

import argparse
import json
import stat
import sys
import tempfile
from pathlib import Path
from typing import Callable

from sysadmin_env.tasks import hpc_munge
from sysadmin_env.tasks import hpc_outage
from sysadmin_env.tasks import hpc_pid_stale


def fix_hpc_outage(root: Path) -> None:
    route_path = root / hpc_outage.COMPUTE_ROUTE_PATH
    route_path.parent.mkdir(parents=True, exist_ok=True)
    route_path.write_text(hpc_outage.FIXED_ROUTE)

    state_path = root / hpc_outage.SHARED_STATE_PATH
    doc = json.loads(state_path.read_text())
    doc["nodes"]["compute-01"]["state"] = "idle"
    doc["nodes"]["compute-01"]["reason"] = ""
    doc["services"]["slurmd@compute-01"] = "active"
    state_path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")


def fix_hpc_munge(root: Path) -> None:
    fix_hpc_outage(root)
    key_path = root / hpc_munge.MUNGE_KEY_PATH
    key_path.write_bytes(hpc_munge.EXPECTED_KEY_BYTES)
    key_path.chmod(hpc_munge.EXPECTED_KEY_MODE)

    state_path = root / hpc_munge.SHARED_STATE_PATH
    doc = json.loads(state_path.read_text())
    doc["services"]["munge@compute-01"] = "active"
    state_path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")


def fix_hpc_pid_stale(root: Path) -> None:
    pid_path = root / hpc_pid_stale.STALE_PID_PATH
    if pid_path.exists():
        pid_path.unlink()

    state_path = root / hpc_pid_stale.SHARED_STATE_PATH
    doc = json.loads(state_path.read_text())
    doc["services"]["slurmd@compute-01"] = "active"
    doc["nodes"]["compute-01"]["state"] = "idle"
    doc["nodes"]["compute-01"]["reason"] = ""
    state_path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")


SCENARIOS: dict[str, tuple[object, Callable[[Path], None]]] = {
    hpc_outage.TASK_ID: (hpc_outage, fix_hpc_outage),
    hpc_munge.TASK_ID: (hpc_munge, fix_hpc_munge),
    hpc_pid_stale.TASK_ID: (hpc_pid_stale, fix_hpc_pid_stale),
}


def verify_scenario(scenario_id: str, *, verbose: bool) -> bool:
    if scenario_id not in SCENARIOS:
        raise KeyError(f"unknown scenario {scenario_id}")
    scenario, fix_fn = SCENARIOS[scenario_id]

    with tempfile.TemporaryDirectory(prefix=f"verify_{scenario_id}_") as tmp:
        root = Path(tmp)
        scenario.prepare_filesystem(root)

        broken = scenario.grade(root)
        if broken.done or broken.health >= 1.0:
            print(f"FAIL {scenario_id} initial state already solved health {broken.health}")
            return False

        if verbose:
            print(f"  {scenario_id} broken  health {broken.health:.2f} details {broken.details}")

        fix_fn(root)
        fixed = scenario.grade(root)
        if not fixed.done or fixed.health < 1.0:
            print(f"FAIL {scenario_id} gold fix did not reach done health {fixed.health} details {fixed.details}")
            return False

        if verbose:
            print(f"  {scenario_id} fixed   health {fixed.health:.2f} details {fixed.details}")

        file_count = sum(1 for _ in root.rglob('*') if _.is_file())
        mode_key = None
        if scenario_id == hpc_munge.TASK_ID:
            key = root / hpc_munge.MUNGE_KEY_PATH
            mode_key = oct(stat.S_IMODE(key.stat().st_mode))
        print(
            f"PASS {scenario_id} files {file_count} "
            f"{'mode_key ' + mode_key if mode_key else ''}".strip()
        )
        return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="+", default=None, help="limit to specific scenario ids")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    targets = args.only or list(SCENARIOS)
    failures: list[str] = []
    for sid in targets:
        ok = verify_scenario(sid, verbose=args.verbose)
        if not ok:
            failures.append(sid)

    if failures:
        print(f"\nFAIL {len(failures)} scenarios unsolved {failures}")
        return 1
    print(f"\nok all {len(targets)} scenarios solvable")
    return 0


if __name__ == "__main__":
    sys.exit(main())
