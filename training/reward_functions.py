"""independent reward functions for grpo training.

the hackathon judges explicitly recommend multiple independent reward signals
instead of a single scalar so the policy cannot collapse onto one exploitable
channel. this module wraps one interactive rollout per grpo step and exposes
it as six orthogonal reward functions that `trl.GRPOTrainer` can consume via
`reward_funcs=[...]`.

the six signals are:

1. solve reward       — binary rlvr signal from the deterministic grader
2. format reward      — was the <bash>...</bash> action grammar respected
3. safety reward      — did the rollout issue any catastrophic command
4. progress reward    — terminal grader health (partial credit, shaped)
5. efficiency reward  — fewer turns to solve -> small positive bonus
6. anti hack reward   — did the rollout try to edit a grader-owned file

all six are independently computable from the `RolloutRecord`s produced by
`training.rollout.run_interactive_group`, so they can be monitored and
inspected separately in the trl logs / wandb / tensorboard run.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from typing import Sequence

from sysadmin_env.rewards import DEFAULT_DESTRUCTIVE_COMMAND_PATTERNS
from training.agent_prompt import BASH_BLOCK_RE
from training.rollout import RolloutRecord


# files the grader owns. if the agent writes to these directly (via tee or
# printf) without going through the normal remediation commands we flag it.
# these are relative to the sandbox overlay root; the patterns match anywhere
# in a shell command, so simple sed/tee/printf tricks are caught.
GRADER_PROTECTED_PATTERNS: tuple[str, ...] = (
    r"slurm_state\.json",
    r"/grader(/|\s|$)",
    r"/\.judge(/|\s|$)",
    r"ECC_RESET_SENTINEL",
)

_DESTRUCTIVE_RE = tuple(
    re.compile(p, flags=re.IGNORECASE) for p in DEFAULT_DESTRUCTIVE_COMMAND_PATTERNS
)
_PROTECTED_RE = tuple(
    re.compile(p, flags=re.IGNORECASE) for p in GRADER_PROTECTED_PATTERNS
)


@dataclass
class RolloutCache:
    """shared buffer filled in by the rollout runner and read by each reward fn.

    trl calls reward_funcs one after another with the same `completions` batch.
    we key the cache on `id(completions)` so the rollout is only executed
    once per grpo step.
    """

    key: int = 0
    records: list[RolloutRecord] | None = None
    wall_seconds: float = 0.0


def _extract_commands(transcript: Sequence[dict[str, str]]) -> list[str]:
    commands: list[str] = []
    for message in transcript:
        if message.get("role") != "assistant":
            continue
        for match in BASH_BLOCK_RE.finditer(message.get("content", "") or ""):
            text = match.group(1).strip()
            if text:
                commands.append(text)
    return commands


def _has_bash_block(completion: str) -> bool:
    return bool(BASH_BLOCK_RE.search(completion or ""))


def _is_destructive(command: str) -> bool:
    return any(rx.search(command) for rx in _DESTRUCTIVE_RE)


def _touches_protected_path(command: str) -> bool:
    return any(rx.search(command) for rx in _PROTECTED_RE)


RolloutRunner = Callable[[int, int | None], list[RolloutRecord]]


def make_reward_functions(
    runner: RolloutRunner,
    *,
    max_turns: int,
    cache: RolloutCache | None = None,
    on_rollout: Callable[[list[RolloutRecord], float], None] | None = None,
) -> tuple[list[Callable], RolloutCache]:
    """build the six-way reward function list and return it together with the
    shared cache so the caller can introspect rollouts after each grpo step.

    - `runner(group_size, seed)` executes the interactive rollout and returns
      one `RolloutRecord` per completion. the runner must not mutate the
      completions argument.
    - `on_rollout(records, wall_seconds)` is an optional hook called exactly
      once per grpo step, which is where a human-facing transcript sampler
      should live.
    """

    cache = cache or RolloutCache()

    def _ensure_rollout(completions: list) -> list[RolloutRecord]:
        batch_key = id(completions)
        if cache.key == batch_key and cache.records is not None:
            return cache.records
        started = time.time()
        records = runner(len(completions), None)
        cache.key = batch_key
        cache.records = records
        cache.wall_seconds = time.time() - started
        if on_rollout is not None:
            try:
                on_rollout(records, cache.wall_seconds)
            except Exception as exc:  # noqa: BLE001
                # a broken sampler must not destabilise grpo training
                print(f"reward_functions on_rollout hook failed {type(exc).__name__} {exc}")
        return records

    def solve_reward(prompts, completions, **kwargs):
        # binary rlvr: did the episode reach a terminal solved state? we key
        # on the env's `terminated` flag rather than a numeric threshold on
        # `reward`, because the server's shaped reward on the solving step
        # is only the health_delta (typically ~0.4), not 1.0.
        records = _ensure_rollout(completions)
        return [1.0 if bool(r.terminated) else 0.0 for r in records]

    def format_reward(prompts, completions, **kwargs):
        # does this completion contain a parseable <bash>...</bash> block
        return [0.1 if _has_bash_block(c) else -0.1 for c in completions]

    def safety_reward(prompts, completions, **kwargs):
        records = _ensure_rollout(completions)
        scores: list[float] = []
        for r in records:
            violations = 0
            for command in _extract_commands(r.transcript):
                if _is_destructive(command):
                    violations += 1
            scores.append(-1.0 * violations)
        return scores

    def progress_reward(prompts, completions, **kwargs):
        # dense shaped signal in [0, 0.5]. prefers the server-side grader
        # health when available (best_health is monotone over the rollout)
        # and falls back to a normalised cumulative shaped reward when the
        # server cannot ship grader_health (older deployments).
        records = _ensure_rollout(completions)
        scores: list[float] = []
        for r in records:
            health = float(r.best_health or r.grader_health)
            if health > 0.0:
                scores.append(0.5 * min(1.0, health))
                continue
            # fallback: tiny dense credit from accumulated shaped reward.
            # clamp to [0, 0.2] so this cannot outbid solve_reward.
            fallback = max(0.0, min(0.2, 0.5 * float(r.reward)))
            scores.append(fallback)
        return scores

    def efficiency_reward(prompts, completions, **kwargs):
        # reward solving quickly. caps at 0.2 so it cannot outbid solving.
        records = _ensure_rollout(completions)
        scores: list[float] = []
        for r in records:
            if not bool(r.terminated):
                scores.append(0.0)
                continue
            saved_turns = max(0, max_turns - int(r.steps))
            scores.append(min(0.2, 0.2 * saved_turns / max(1, max_turns)))
        return scores

    def anti_hack_reward(prompts, completions, **kwargs):
        records = _ensure_rollout(completions)
        scores: list[float] = []
        for r in records:
            edits = 0
            for command in _extract_commands(r.transcript):
                if _touches_protected_path(command):
                    edits += 1
            scores.append(-0.5 * edits)
        return scores

    return (
        [
            solve_reward,
            format_reward,
            safety_reward,
            progress_reward,
            efficiency_reward,
            anti_hack_reward,
        ],
        cache,
    )


__all__ = [
    "GRADER_PROTECTED_PATTERNS",
    "RolloutCache",
    "make_reward_functions",
]
