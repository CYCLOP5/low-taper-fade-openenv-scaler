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
        transcript_sample_every: int = 5,
        transcript_max_samples: int = 2,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_name = run_name
        self.jsonl_path = self.output_dir / f"{run_name}.metrics.jsonl"
        self.transcripts_dir = self.output_dir / "transcripts"
        self.transcripts_dir.mkdir(parents=True, exist_ok=True)
        self.transcript_sample_every = max(1, int(transcript_sample_every))
        self.transcript_max_samples = max(1, int(transcript_max_samples))
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
        # judges' guide: "sample outputs frequently and inspect them". write a
        # couple of transcripts to disk every few steps so reward hacking is
        # catchable by a human reviewer and so tensorboard text panels have
        # something to show.
        if step % self.transcript_sample_every == 0:
            self._write_transcript_sample(step, records)
        return metrics

    def _write_transcript_sample(self, step: int, records: list[Any]) -> None:
        if not records:
            return
        sample_path = self.transcripts_dir / f"step_{step:05d}.jsonl"
        with sample_path.open("w") as f:
            for r in records[: self.transcript_max_samples]:
                transcript = getattr(r, "transcript", None) or []
                payload = {
                    "task_id": getattr(r, "task_id", ""),
                    "reward": float(getattr(r, "reward", 0.0)),
                    "steps": int(getattr(r, "steps", 0)),
                    "grader_health": float(getattr(r, "grader_health", 0.0)),
                    "terminated": bool(getattr(r, "terminated", False)),
                    "truncated": bool(getattr(r, "truncated", False)),
                    "transcript": transcript,
                }
                f.write(json.dumps(payload, default=str) + "\n")

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
