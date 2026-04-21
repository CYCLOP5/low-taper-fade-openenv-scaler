from __future__ import annotations

import statistics
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Callable
from typing import Sequence

from hpc_gym import EnterpriseHPCEnv
from training.agent_prompt import SYSTEM_PROMPT
from training.agent_prompt import USER_PROMPT
from training.agent_prompt import iter_actions
from training.agent_prompt import parse_action


GenerateFn = Callable[[list[list[dict[str, str]]]], list[str]]


@dataclass
class RolloutRecord:
    reward: float
    steps: int
    terminated: bool
    truncated: bool
    task_id: str
    transcript: list[dict[str, str]] = field(default_factory=list)
    grader_health: float = 0.0
    ood_http_code: str = ""


def score_single_shot(completions: Sequence[str], env: EnterpriseHPCEnv) -> list[RolloutRecord]:
    records: list[RolloutRecord] = []
    for completion in completions:
        env.reset()
        reward = 0.0
        health = 0.0
        http_code = ""
        steps = 0
        terminated = False
        truncated = False
        task_id = env.scenario.TASK_ID
        for action in iter_actions(completion):
            _, reward, terminated, truncated, info = env.step(action)
            steps += 1
            health = float(info.get("grader_health", 0.0))
            http_code = str(info.get("ood_http_code", ""))
            if terminated or truncated:
                break
        records.append(
            RolloutRecord(
                reward=reward,
                steps=steps,
                terminated=terminated,
                truncated=truncated,
                task_id=task_id,
                grader_health=health,
                ood_http_code=http_code,
            )
        )
    return records


def run_interactive_group(
    group_size: int,
    generate_fn: GenerateFn,
    env_factory: Callable[[], EnterpriseHPCEnv],
    max_turns: int,
    seed_start: int = 0,
) -> list[RolloutRecord]:
    envs: list[EnterpriseHPCEnv] = []
    transcripts: list[list[dict[str, str]]] = []
    observations: list[str] = []
    done: list[bool] = []
    rewards: list[float] = [0.0] * group_size
    health: list[float] = [0.0] * group_size
    http_codes: list[str] = [""] * group_size
    steps_taken: list[int] = [0] * group_size
    terminated_list: list[bool] = [False] * group_size
    truncated_list: list[bool] = [False] * group_size
    task_ids: list[str] = [""] * group_size

    for idx in range(group_size):
        env = env_factory()
        obs, info = env.reset(seed=seed_start + idx)
        envs.append(env)
        task_ids[idx] = info.get("task_id") or getattr(getattr(env, "scenario", None), "TASK_ID", "") or ""
        transcripts.append(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"{USER_PROMPT}\n\ncurrent observation:\n{obs}"},
            ]
        )
        observations.append(obs)
        done.append(False)

    try:
        for _ in range(max_turns):
            active = [i for i in range(group_size) if not done[i]]
            if not active:
                break
            batch = [transcripts[i] for i in active]
            completions = generate_fn(batch)
            if len(completions) != len(active):
                raise RuntimeError(
                    f"generate_fn returned {len(completions)} completions expected {len(active)}"
                )
            for j, idx in enumerate(active):
                completion = completions[j]
                transcripts[idx].append({"role": "assistant", "content": completion})

                command, voluntary_done = parse_action(completion)
                if command is None:
                    transcripts[idx].append(
                        {
                            "role": "user",
                            "content": "error no bash block detected. emit exactly one <bash>...</bash> block",
                        }
                    )
                    steps_taken[idx] += 1
                    if steps_taken[idx] >= max_turns:
                        truncated_list[idx] = True
                        done[idx] = True
                    continue

                obs, reward, terminated, truncated, info = envs[idx].step(command)
                steps_taken[idx] += 1
                rewards[idx] = reward
                health[idx] = float(info.get("grader_health", 0.0))
                http_codes[idx] = str(info.get("ood_http_code", ""))
                terminated_list[idx] = bool(terminated)
                truncated_list[idx] = bool(truncated)

                if voluntary_done or terminated or truncated:
                    done[idx] = True
                else:
                    transcripts[idx].append(
                        {
                            "role": "user",
                            "content": f"step {steps_taken[idx]} observation:\n{obs}",
                        }
                    )
    finally:
        for env in envs:
            try:
                env.close()
            except Exception:
                pass

    return [
        RolloutRecord(
            reward=rewards[i],
            steps=steps_taken[i],
            terminated=terminated_list[i],
            truncated=truncated_list[i],
            task_id=task_ids[i],
            transcript=transcripts[i],
            grader_health=health[i],
            ood_http_code=http_codes[i],
        )
        for i in range(group_size)
    ]


def summarize_group(records: Sequence[RolloutRecord]) -> dict[str, float]:
    if not records:
        return {}
    rewards = [r.reward for r in records]
    steps = [r.steps for r in records]
    solved = sum(1 for r in records if r.reward >= 1.0)
    return {
        "n": float(len(records)),
        "reward_mean": statistics.fmean(rewards),
        "reward_max": max(rewards),
        "solve_rate": solved / len(records),
        "steps_mean": statistics.fmean(steps),
        "health_mean": statistics.fmean(r.grader_health for r in records),
    }


def run_fixed_policy(
    env: EnterpriseHPCEnv,
    actions: Sequence[str],
    reset_options: dict[str, Any] | None = None,
) -> RolloutRecord:
    obs, info = env.reset(options=reset_options)
    reward = 0.0
    steps = 0
    terminated = False
    truncated = False
    task_id = info.get("task_id") or getattr(getattr(env, "scenario", None), "TASK_ID", "") or ""
    health = 0.0
    http_code = ""
    transcript = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{USER_PROMPT}\n\ncurrent observation:\n{obs}"},
    ]
    for command in actions:
        transcript.append({"role": "assistant", "content": f"<bash>{command}</bash>"})
        obs, reward, terminated, truncated, info = env.step(command)
        steps += 1
        health = float(info.get("grader_health", 0.0))
        http_code = str(info.get("ood_http_code", ""))
        transcript.append({"role": "user", "content": f"step {steps} observation:\n{obs}"})
        if terminated or truncated:
            break
    return RolloutRecord(
        reward=reward,
        steps=steps,
        terminated=terminated,
        truncated=truncated,
        task_id=task_id,
        transcript=transcript,
        grader_health=health,
        ood_http_code=http_code,
    )
