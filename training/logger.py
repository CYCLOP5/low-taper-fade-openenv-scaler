from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class StepMetrics:
    step: int
    solve_rate: float
    reward_mean: float
    reward_max: float
    health_mean: float
    steps_mean: float
    task_mix: dict[str, int]
    wall_seconds: float


class RewardLogger:
    def __init__(
        self,
        output_dir: str | Path,
        run_name: str = "hpc_grpo",
        wandb_project: str | None = None,
        hub_repo: str | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_name = run_name
        self.jsonl_path = self.output_dir / f"{run_name}.metrics.jsonl"
        self._start = time.time()
        self._wandb = None
        if wandb_project:
            try:
                import wandb  # type: ignore

                self._wandb = wandb.init(
                    project=wandb_project,
                    name=run_name,
                    dir=str(self.output_dir),
                    reinit=True,
                )
            except Exception as exc:
                print(f"reward_logger wandb disabled {type(exc).__name__.lower()} {exc}")
                self._wandb = None
        self.hub_repo = hub_repo

    def log(self, step: int, records: list[Any]) -> StepMetrics:
        rewards = [float(r.reward) for r in records]
        health = [float(r.grader_health) for r in records]
        steps = [int(r.steps) for r in records]
        solved = sum(1 for r in records if r.reward >= 1.0)
        mix: dict[str, int] = {}
        for r in records:
            mix[r.task_id] = mix.get(r.task_id, 0) + 1
        metrics = StepMetrics(
            step=step,
            solve_rate=solved / len(records) if records else 0.0,
            reward_mean=(sum(rewards) / len(rewards)) if rewards else 0.0,
            reward_max=max(rewards) if rewards else 0.0,
            health_mean=(sum(health) / len(health)) if health else 0.0,
            steps_mean=(sum(steps) / len(steps)) if steps else 0.0,
            task_mix=mix,
            wall_seconds=time.time() - self._start,
        )
        payload = asdict(metrics)
        with self.jsonl_path.open("a") as f:
            f.write(json.dumps(payload) + "\n")
        if self._wandb is not None:
            try:
                self._wandb.log(payload, step=step)
            except Exception as exc:
                print(f"reward_logger wandb log failed {type(exc).__name__.lower()} {exc}")
        print(
            f"metrics step {step} solve_rate {metrics.solve_rate:.2f} "
            f"reward_mean {metrics.reward_mean:.2f} health_mean {metrics.health_mean:.2f} "
            f"steps_mean {metrics.steps_mean:.1f} mix {mix}"
        )
        return metrics

    def close(self) -> None:
        if self._wandb is not None:
            try:
                self._wandb.finish()
            except Exception:
                pass
        if self.hub_repo:
            self._push_to_hub()

    def _push_to_hub(self) -> None:
        try:
            from huggingface_hub import HfApi  # type: ignore

            api = HfApi(token=os.environ.get("HF_TOKEN"))
            api.upload_file(
                path_or_fileobj=str(self.jsonl_path),
                path_in_repo=f"runs/{self.jsonl_path.name}",
                repo_id=self.hub_repo,
                repo_type="model",
            )
            print(f"reward_logger pushed metrics to hub {self.hub_repo}")
        except Exception as exc:
            print(f"reward_logger hub push failed {type(exc).__name__.lower()} {exc}")
