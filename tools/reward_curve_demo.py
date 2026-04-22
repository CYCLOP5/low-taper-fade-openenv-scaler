from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from sysadmin_env.rewards import build_reward_engine
from sysadmin_env.tasks import build_task_registry
from sysadmin_env.tasks import hpc_gpu_ecc
from sysadmin_env.tasks import hpc_munge
from sysadmin_env.tasks import hpc_nfs_stale
from sysadmin_env.tasks import hpc_ood_apache
from sysadmin_env.tasks import hpc_outage
from sysadmin_env.tasks import hpc_pid_stale
from tools import verify_gold_trajectory as gold


# each action primitive is a function (root) -> None that performs a small portion
# of a scenario fix. the demo agent samples commands by name and we materialise the
# deterministic side effect so the grader sees meaningful partial progress without
# needing a live sandbox. this models the reward landscape the real trl + gemma 4
# agent navigates at each grpo step.


@dataclass
class Primitive:
    command: str
    scenarios: set[str]
    apply: Callable[[Path], None]


def _write_state(path: Path, mutator: Callable[[dict], None]) -> None:
    doc = json.loads(path.read_text()) if path.exists() else {}
    mutator(doc)
    path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")


def _fix_route(root: Path) -> None:
    path = root / hpc_outage.COMPUTE_ROUTE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(hpc_outage.FIXED_ROUTE)


def _restart_slurmd_if_route_fixed(root: Path) -> None:
    state_path = root / hpc_outage.SHARED_STATE_PATH
    route_path = root / hpc_outage.COMPUTE_ROUTE_PATH

    def apply(doc: dict) -> None:
        route_ok = route_path.exists() and route_path.read_text() == hpc_outage.FIXED_ROUTE
        mount_ok = (root / hpc_nfs_stale.MOUNT_VALID_RELATIVE).exists() and not (
            root / hpc_nfs_stale.MOUNT_STALE_RELATIVE
        ).exists()
        ecc_ok = (root / hpc_gpu_ecc.ECC_RESET_PATH).exists()
        munge_key = root / hpc_munge.MUNGE_KEY_PATH
        munge_ok = (
            munge_key.exists()
            and munge_key.read_bytes() == hpc_munge.EXPECTED_KEY_BYTES
            and (munge_key.stat().st_mode & 0o777) == hpc_munge.EXPECTED_KEY_MODE
        )
        pid_ok = not (root / hpc_pid_stale.STALE_PID_PATH).exists()

        services = doc.setdefault("services", {})
        nodes = doc.setdefault("nodes", {})
        compute = nodes.setdefault("compute-01", {})
        if route_ok and mount_ok and ecc_ok and munge_ok and pid_ok:
            services["slurmd@compute-01"] = "active"
            compute["state"] = "idle"
            compute["reason"] = ""

    _write_state(state_path, apply)


def _fix_munge_contents(root: Path) -> None:
    path = root / hpc_munge.MUNGE_KEY_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.chmod(0o600)
    path.write_bytes(hpc_munge.EXPECTED_KEY_BYTES)


def _fix_munge_mode(root: Path) -> None:
    path = root / hpc_munge.MUNGE_KEY_PATH
    if path.exists():
        path.chmod(hpc_munge.EXPECTED_KEY_MODE)


def _restart_munge(root: Path) -> None:
    path = root / hpc_munge.MUNGE_KEY_PATH
    if (
        path.exists()
        and path.read_bytes() == hpc_munge.EXPECTED_KEY_BYTES
        and (path.stat().st_mode & 0o777) == hpc_munge.EXPECTED_KEY_MODE
    ):
        def apply(doc: dict) -> None:
            doc.setdefault("services", {})["munge@compute-01"] = "active"
        _write_state(root / hpc_munge.SHARED_STATE_PATH, apply)


def _remove_stale_pid(root: Path) -> None:
    path = root / hpc_pid_stale.STALE_PID_PATH
    if path.exists():
        path.unlink()


def _reset_gpu_ecc(root: Path) -> None:
    sentinel = root / hpc_gpu_ecc.ECC_RESET_PATH
    sentinel.parent.mkdir(parents=True, exist_ok=True)
    sentinel.write_text("reset ok\n")

    def apply(doc: dict) -> None:
        gpus = doc.setdefault("gpus", {})
        entry = gpus.setdefault("compute-01:gpu-0", {})
        entry["state"] = "healthy"
        entry["ecc_vol_total"] = 0

    _write_state(root / hpc_gpu_ecc.SHARED_STATE_PATH, apply)


def _umount_shared(root: Path) -> None:
    stale = root / hpc_nfs_stale.MOUNT_STALE_RELATIVE
    if stale.exists():
        stale.unlink()


def _mount_shared(root: Path) -> None:
    stale = root / hpc_nfs_stale.MOUNT_STALE_RELATIVE
    valid = root / hpc_nfs_stale.MOUNT_VALID_RELATIVE
    if not stale.exists() and not valid.exists():
        valid.parent.mkdir(parents=True, exist_ok=True)
        valid.write_text("fresh mount handle\n")


def _fix_httpd_conf(root: Path) -> None:
    path = root / hpc_ood_apache.HTTPD_CONF_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(hpc_ood_apache.FIXED_HTTPD_CONF)


def _apachectl_graceful(root: Path) -> None:
    conf = root / hpc_ood_apache.HTTPD_CONF_PATH
    if conf.exists() and conf.read_text() == hpc_ood_apache.FIXED_HTTPD_CONF:
        def apply(doc: dict) -> None:
            doc.setdefault("services", {})["httpd@login"] = "active"
            portals = doc.setdefault("portals", {})
            ood = portals.setdefault("apache_ood", {})
            ood["state"] = "healthy"
            ood["last_error"] = ""

        _write_state(root / hpc_ood_apache.SHARED_STATE_PATH, apply)


def _noop(_: Path) -> None:
    return None


ALL_SCENARIOS = {
    hpc_outage.TASK_ID,
    hpc_munge.TASK_ID,
    hpc_pid_stale.TASK_ID,
    hpc_gpu_ecc.TASK_ID,
    hpc_nfs_stale.TASK_ID,
    hpc_ood_apache.TASK_ID,
}


PRIMITIVES: list[Primitive] = [
    Primitive("sinfo", ALL_SCENARIOS, _noop),
    Primitive("squeue", ALL_SCENARIOS, _noop),
    Primitive("ssh compute-01", ALL_SCENARIOS, _noop),
    Primitive(
        "printf '<FIXED_ROUTE>' > /etc/sysconfig/network-scripts/route-eth0",
        {hpc_outage.TASK_ID, hpc_munge.TASK_ID},
        _fix_route,
    ),
    Primitive(
        "chmod 0400 /etc/munge/munge.key",
        {hpc_munge.TASK_ID},
        _fix_munge_mode,
    ),
    Primitive(
        "printf '<MUNGE_KEY>' > /etc/munge/munge.key",
        {hpc_munge.TASK_ID},
        _fix_munge_contents,
    ),
    Primitive(
        "systemctl restart munge",
        {hpc_munge.TASK_ID},
        _restart_munge,
    ),
    Primitive(
        "rm /var/run/slurmd.pid",
        {hpc_pid_stale.TASK_ID},
        _remove_stale_pid,
    ),
    Primitive(
        "nvidia-smi -r -i 0",
        {hpc_gpu_ecc.TASK_ID},
        _reset_gpu_ecc,
    ),
    Primitive(
        "umount -l /mnt/shared",
        {hpc_nfs_stale.TASK_ID},
        _umount_shared,
    ),
    Primitive(
        "mount /mnt/shared",
        {hpc_nfs_stale.TASK_ID},
        _mount_shared,
    ),
    Primitive(
        "printf '<HTTPD_CONF>' > /etc/httpd/conf/httpd.conf",
        {hpc_ood_apache.TASK_ID},
        _fix_httpd_conf,
    ),
    Primitive(
        "apachectl graceful",
        {hpc_ood_apache.TASK_ID},
        _apachectl_graceful,
    ),
    Primitive(
        "systemctl restart slurmd",
        ALL_SCENARIOS - {hpc_ood_apache.TASK_ID},
        _restart_slurmd_if_route_fixed,
    ),
    Primitive(
        "curl -I http://localhost:8080/",
        ALL_SCENARIOS,
        _noop,
    ),
    Primitive(
        "curl -I http://localhost:8081/",
        {hpc_ood_apache.TASK_ID},
        _noop,
    ),
    Primitive("cat /etc/hostname", ALL_SCENARIOS, _noop),
    Primitive("ls /mnt/shared", ALL_SCENARIOS, _noop),
    Primitive("echo noise > /tmp/distraction", set(), _noop),
    Primitive("true", set(), _noop),
]


GOLD_HINTS: dict[str, list[str]] = {
    hpc_outage.TASK_ID: [
        "printf '<FIXED_ROUTE>' > /etc/sysconfig/network-scripts/route-eth0",
        "systemctl restart slurmd",
    ],
    hpc_munge.TASK_ID: [
        "printf '<MUNGE_KEY>' > /etc/munge/munge.key",
        "chmod 0400 /etc/munge/munge.key",
        "systemctl restart munge",
        "systemctl restart slurmd",
    ],
    hpc_pid_stale.TASK_ID: [
        "rm /var/run/slurmd.pid",
        "systemctl restart slurmd",
    ],
    hpc_gpu_ecc.TASK_ID: [
        "nvidia-smi -r -i 0",
        "systemctl restart slurmd",
    ],
    hpc_nfs_stale.TASK_ID: [
        "umount -l /mnt/shared",
        "mount /mnt/shared",
        "systemctl restart slurmd",
    ],
    hpc_ood_apache.TASK_ID: [
        "printf '<HTTPD_CONF>' > /etc/httpd/conf/httpd.conf",
        "apachectl graceful",
    ],
}


def _sample_policy(
    rng: random.Random,
    scenario: str,
    temperature: float,
    max_steps: int,
) -> list[Primitive]:
    # temperature=1.0 -> uniform random. temperature=0.0 -> always gold primitives
    # for this scenario. in between we interpolate probabilities, mimicking what a
    # model learns over grpo steps as the log policy sharpens around rewarding
    # actions.
    relevant = [p for p in PRIMITIVES if scenario in p.scenarios]
    irrelevant = [p for p in PRIMITIVES if scenario not in p.scenarios]
    good_weight = max(1e-4, 1.0 - temperature) * 6.0 + 0.5
    bad_weight = temperature * 2.0 + 0.1

    weights = [good_weight] * len(relevant) + [bad_weight] * len(irrelevant)
    pool = relevant + irrelevant
    actions = [rng.choices(pool, weights=weights, k=1)[0] for _ in range(max_steps)]

    # at low temperature inject the gold hints at the front so the simulated policy
    # behaves like a well-trained agent that has internalised the gold trajectory
    gold_prob = max(0.0, 1.0 - temperature * 1.4)
    if gold_prob > 0.0 and rng.random() < gold_prob:
        hint_commands = GOLD_HINTS.get(scenario, [])
        by_command = {p.command: p for p in PRIMITIVES}
        prefix = [by_command[c] for c in hint_commands if c in by_command]
        actions = prefix + actions
        actions = actions[:max_steps]
    return actions


def _run_episode(
    scenario: str,
    registry: dict,
    engine,
    policy: list[Primitive],
    max_steps: int,
) -> dict:
    module = {
        hpc_outage.TASK_ID: hpc_outage,
        hpc_munge.TASK_ID: hpc_munge,
        hpc_pid_stale.TASK_ID: hpc_pid_stale,
        hpc_gpu_ecc.TASK_ID: hpc_gpu_ecc,
        hpc_nfs_stale.TASK_ID: hpc_nfs_stale,
        hpc_ood_apache.TASK_ID: hpc_ood_apache,
    }[scenario]

    with tempfile.TemporaryDirectory(prefix=f"demo_{scenario}_") as tmp:
        root = Path(tmp)
        module.prepare_filesystem(root)

        state = engine.start_episode(scenario, runtime_root=root)
        shaped_total = 0.0
        terminal_health = 0.0
        steps = 0
        solved = False
        for primitive in policy:
            steps += 1
            primitive.apply(root)
            computation = engine.evaluate_action(state, primitive.command)
            shaped_total += computation.signal.total_reward
            terminal_health = computation.task_state.health
            if computation.task_state.done:
                solved = True
                break
            if computation.catastrophic:
                break
            if steps >= max_steps:
                break
        return {
            "scenario": scenario,
            "steps": steps,
            "shaped_reward_sum": shaped_total,
            "terminal_health": terminal_health,
            "solved": bool(solved),
        }


def _run_step(
    rng: random.Random,
    engine,
    registry: dict,
    scenarios: list[str],
    temperature: float,
    rollouts: int,
    max_steps: int,
) -> dict:
    rows: list[dict] = []
    for _ in range(rollouts):
        scenario = rng.choice(scenarios)
        policy = _sample_policy(rng, scenario, temperature, max_steps)
        rows.append(_run_episode(scenario, registry, engine, policy, max_steps))
    shaped = [row["shaped_reward_sum"] for row in rows]
    solved = [1.0 if row["solved"] else 0.0 for row in rows]
    health = [row["terminal_health"] for row in rows]
    return {
        "rollouts": rows,
        "temperature": temperature,
        "reward_mean": statistics.fmean(shaped),
        "reward_std": statistics.pstdev(shaped) if len(shaped) > 1 else 0.0,
        "solve_rate": statistics.fmean(solved),
        "terminal_health_mean": statistics.fmean(health),
    }


def _plot_curve(history: list[dict], output_path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable skipping plot artifact write")
        return

    steps = [row["step"] for row in history]
    reward = [row["reward_mean"] for row in history]
    solve = [row["solve_rate"] for row in history]
    health = [row["terminal_health_mean"] for row in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    axes[0].plot(steps, reward, label="shaped reward mean", color="#2a6f97")
    axes[0].plot(steps, health, label="terminal health mean", color="#7fb069", linestyle="--")
    axes[0].set_title("reward curve over simulated curriculum steps")
    axes[0].set_xlabel("curriculum step")
    axes[0].set_ylabel("reward / health")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="lower right")

    axes[1].plot(steps, solve, label="solve rate", color="#d62828")
    axes[1].set_title("solve rate over simulated curriculum steps")
    axes[1].set_xlabel("curriculum step")
    axes[1].set_ylabel("solve rate")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="lower right")

    fig.suptitle("EnterpriseHPC-v0 reward landscape probe (gpu free)")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "curriculum-annealed reward probe for EnterpriseHPC-v0 that runs without "
            "bwrap or a gpu. it generates a reward curve png that proves the shaped "
            "reward signal has a meaningful gradient an agent can climb."
        ),
    )
    parser.add_argument(
        "--scenarios",
        default="hpc_outage,hpc_munge,hpc_pid_stale,hpc_gpu_ecc,hpc_nfs_stale,hpc_ood_apache",
    )
    parser.add_argument("--num-steps", type=int, default=24)
    parser.add_argument("--rollouts-per-step", type=int, default=12)
    parser.add_argument("--max-actions", type=int, default=14)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--output-dir", default="./runs/reward_demo")
    parser.add_argument(
        "--plot-path",
        default="docs/assets/reward_curve_demo.png",
        help="png path relative to the repo root",
    )
    args = parser.parse_args(argv)

    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    rng = random.Random(args.seed)

    with tempfile.TemporaryDirectory(prefix="reward_demo_registry_") as tmp:
        registry = build_task_registry(tmp)
        engine = build_reward_engine(registry)

        history: list[dict] = []
        for step in range(args.num_steps):
            progress = step / max(args.num_steps - 1, 1)
            temperature = max(0.05, 1.0 - 0.93 * progress)
            summary = _run_step(
                rng=rng,
                engine=engine,
                registry=registry,
                scenarios=scenarios,
                temperature=temperature,
                rollouts=args.rollouts_per_step,
                max_steps=args.max_actions,
            )
            summary["step"] = step
            history.append(summary)
            print(
                f"step {step:>3} temp {temperature:.2f} "
                f"reward_mean {summary['reward_mean']:.3f} "
                f"solve_rate {summary['solve_rate']:.2f} "
                f"terminal_health {summary['terminal_health_mean']:.2f}"
            )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "reward_curve.jsonl"
    metrics_path.write_text(
        "\n".join(
            json.dumps(
                {k: v for k, v in row.items() if k != "rollouts"},
                sort_keys=True,
            )
            for row in history
        )
        + "\n"
    )
    plot_path = Path(args.plot_path)
    _plot_curve(history, plot_path)
    print(f"metrics {metrics_path}")
    print(f"plot {plot_path}")

    # sanity check at least one scenario improved meaningfully
    start = history[0]["reward_mean"]
    end = history[-1]["reward_mean"]
    delta = end - start
    if delta < 0.05:
        print(
            f"warning reward delta {delta:.3f} seems small which may indicate the "
            "shaping changed rerun with more rollouts or check the primitive map"
        )
    else:
        print(f"ok reward improvement {delta:.3f} over {args.num_steps} steps")

    # prove gold trajectory still solves each scenario end to end
    gold_failures: list[str] = []
    for scenario_id in scenarios:
        if not gold.verify_scenario(scenario_id, verbose=False):
            gold_failures.append(scenario_id)
    if gold_failures:
        print(f"fail gold trajectory regression {gold_failures}")
        return 1
    print("ok gold trajectory still solvable for all scenarios")
    return 0


if __name__ == "__main__":
    sys.exit(main())
