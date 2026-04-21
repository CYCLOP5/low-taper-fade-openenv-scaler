from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable

from sysadmin_env.tasks import hpc_outage


GOLD_TRAJECTORY_OUTAGE = [
    "sinfo",
    "squeue",
    "ssh compute-01",
    "cat /etc/sysconfig/network-scripts/route-eth0",
    f"printf '{hpc_outage.FIXED_ROUTE}' > /etc/sysconfig/network-scripts/route-eth0",
    "systemctl restart slurmd",
    "exit",
    "curl -I http://localhost:8080/",
]

GOLD_TRAJECTORY_MUNGE = [
    "sinfo",
    "ssh compute-01",
    "ls -l /etc/munge/munge.key",
    f"printf '{hpc_outage.FIXED_ROUTE}' > /etc/sysconfig/network-scripts/route-eth0",
    "chmod 0400 /etc/munge/munge.key",
    "systemctl restart munge",
    "systemctl restart slurmd",
    "exit",
    "curl -I http://localhost:8080/",
]

GOLD_TRAJECTORY_PID_STALE = [
    "sinfo",
    "squeue",
    "ssh compute-01",
    "systemctl status slurmd",
    "cat /var/run/slurmd.pid",
    "rm /var/run/slurmd.pid",
    "systemctl restart slurmd",
    "exit",
    "curl -I http://localhost:8080/",
]

RANDOM_POOL = [
    "sinfo",
    "squeue",
    "ssh compute-01",
    "cat /etc/sysconfig/network-scripts/route-eth0",
    f"printf '{hpc_outage.FIXED_ROUTE}' > /etc/sysconfig/network-scripts/route-eth0",
    "echo garbage > /etc/sysconfig/network-scripts/route-eth0",
    "systemctl restart slurmd",
    "systemctl restart munge",
    "chmod 0400 /etc/munge/munge.key",
    "chmod 0777 /etc/munge/munge.key",
    "cat /var/run/slurmd.pid",
    "rm /var/run/slurmd.pid",
    "ls /mnt/shared",
    "exit",
    "curl -I http://localhost:8080/",
]

BAD_TRAJECTORY = [
    "sinfo",
    "squeue",
    "ls -la /mnt/shared",
    "cat /etc/hostname",
    "exit",
]


def _env_factory(env_urls: list[str] | None, scenarios: list[str]) -> Callable:
    if env_urls:
        from training.remote_env import HttpEnterpriseHPCEnv
        from training.remote_env import RemoteEndpointPool

        pool = RemoteEndpointPool(env_urls)

        def make_env():
            return HttpEnterpriseHPCEnv(env_urls=env_urls, scenario_pool=scenarios, pool=pool)

        return make_env

    from hpc_gym import EnterpriseHPCEnv

    def make_env():
        return EnterpriseHPCEnv(scenario_pool=scenarios)

    return make_env


def _run_policy(
    name: str,
    make_env: Callable,
    scenarios: list[str],
    actions_for: Callable[[str, random.Random], list[str]],
    trials: int,
    seed: int,
) -> list[dict]:
    from training.rollout import run_fixed_policy

    rng = random.Random(seed)
    rows: list[dict] = []
    for scenario in scenarios:
        for trial in range(trials):
            env = make_env()
            try:
                actions = actions_for(scenario, rng)
                record = run_fixed_policy(env, actions, reset_options={"scenario": scenario})
                rows.append(
                    {
                        "policy": name,
                        "scenario": scenario,
                        "trial": trial,
                        "reward": record.reward,
                        "steps": record.steps,
                        "terminated": record.terminated,
                        "grader_health": record.grader_health,
                        "ood_http_code": record.ood_http_code,
                        "task_id": record.task_id,
                    }
                )
            finally:
                try:
                    env.close()
                except Exception:
                    pass
    return rows


def _summarize(rows: list[dict]) -> dict:
    buckets: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        key = (row["policy"], row["scenario"])
        buckets.setdefault(key, []).append(row)
    summary: list[dict] = []
    for (policy, scenario), items in sorted(buckets.items()):
        rewards = [i["reward"] for i in items]
        summary.append(
            {
                "policy": policy,
                "scenario": scenario,
                "n": len(items),
                "solve_rate": sum(1 for r in rewards if r >= 1.0) / len(rewards),
                "reward_mean": statistics.fmean(rewards),
                "steps_mean": statistics.fmean(i["steps"] for i in items),
                "health_mean": statistics.fmean(i["grader_health"] for i in items),
            }
        )
    return {"rows": rows, "summary": summary}


def _write_markdown(path: Path, summary: dict) -> None:
    lines = [
        "# EnterpriseHPC-v0 eval leaderboard",
        "",
        "| policy | scenario | n | solve_rate | reward_mean | steps_mean | health_mean |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["summary"]:
        lines.append(
            f"| {row['policy']} | {row['scenario']} | {row['n']} | "
            f"{row['solve_rate']:.2f} | {row['reward_mean']:.2f} | "
            f"{row['steps_mean']:.1f} | {row['health_mean']:.2f} |"
        )
    lines.append("")
    lines.append(f"_generated_: unix_{int(time.time())}")
    path.write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--scenarios", default="hpc_outage,hpc_munge,hpc_pid_stale")
    parser.add_argument("--policies", default="gold,random,bad")
    parser.add_argument("--env-urls", nargs="+", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="./runs/eval")
    args = parser.parse_args()

    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    policies = [p.strip() for p in args.policies.split(",") if p.strip()]

    make_env = _env_factory(args.env_urls, scenarios)

    def gold_actions(scenario: str, _: random.Random) -> list[str]:
        if scenario == "hpc_munge":
            return GOLD_TRAJECTORY_MUNGE
        if scenario == "hpc_pid_stale":
            return GOLD_TRAJECTORY_PID_STALE
        return GOLD_TRAJECTORY_OUTAGE

    def random_actions(_: str, rng: random.Random) -> list[str]:
        return [rng.choice(RANDOM_POOL) for _ in range(12)]

    def bad_actions(_: str, __: random.Random) -> list[str]:
        return BAD_TRAJECTORY

    policy_fns = {"gold": gold_actions, "random": random_actions, "bad": bad_actions}

    rows: list[dict] = []
    for policy in policies:
        if policy not in policy_fns:
            print(f"unknown policy {policy} skipping", file=sys.stderr)
            continue
        rows.extend(
            _run_policy(
                name=policy,
                make_env=make_env,
                scenarios=scenarios,
                actions_for=policy_fns[policy],
                trials=args.trials,
                seed=args.seed + hash(policy) % 997,
            )
        )

    summary = _summarize(rows)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "eval.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (out / "eval_summary.json").write_text(json.dumps(summary, indent=2))
    _write_markdown(out / "leaderboard.md", summary)

    for row in summary["summary"]:
        print(
            f"{row['policy']:<8} {row['scenario']:<12} n={row['n']:<3} "
            f"solve={row['solve_rate']:.2f} reward={row['reward_mean']:.2f} "
            f"steps={row['steps_mean']:.1f} health={row['health_mean']:.2f}"
        )
    print(f"\nartifacts written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
