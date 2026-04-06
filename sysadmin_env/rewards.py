from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from sysadmin_env.models import RewardSignal
from sysadmin_env.models import TaskScenarioDefinition
from sysadmin_env.models import TaskScenarioState
from sysadmin_env.tasks import get_task_module


DEFAULT_STEP_PENALTY = -0.01
DEFAULT_CATASTROPHIC_PENALTY = -1.0
DEFAULT_DESTRUCTIVE_COMMAND_PATTERNS = (
    r"(^|\s)rm\s+-rf\s+/($|\s)",
    r"(^|\s)rm\s+-rf\s+--no-preserve-root($|\s)",
    r"(^|\s)mkfs(\.|\s|$)",
    r"(^|\s)shutdown(\s|$)",
    r"(^|\s)reboot(\s|$)",
    r"(^|\s)halt(\s|$)",
    r"(^|\s)kill\s+(-9\s+)?1($|\s)",
    r"(^|\s)(dd|truncate)\b.*(of=|>)\s*/(etc|boot)(/|\s|$)",
    r":\s*\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:",
)


@dataclass
class EpisodeRewardState:
    task_id: str
    runtime_root: str
    known_fact_ids: set[str]
    last_health: float
    done: bool


@dataclass
class RewardComputation:
    signal: RewardSignal
    state: EpisodeRewardState
    task_state: TaskScenarioState
    catastrophic: bool


class RewardEngine:
    def __init__(
        self,
        task_registry: dict[str, TaskScenarioDefinition],
        step_penalty: float = DEFAULT_STEP_PENALTY,
        catastrophic_penalty: float = DEFAULT_CATASTROPHIC_PENALTY,
        destructive_command_patterns: tuple[str, ...] = DEFAULT_DESTRUCTIVE_COMMAND_PATTERNS,
    ) -> None:
        self.task_registry = task_registry
        self.step_penalty = step_penalty
        self.catastrophic_penalty = catastrophic_penalty
        self.destructive_command_patterns = tuple(destructive_command_patterns)

    def start_episode(self, task_id: str, runtime_root: str | Path | None = None) -> EpisodeRewardState:
        definition = self.task_registry[task_id]
        effective_root = Path(runtime_root or definition.metadata.base_filesystem_path)
        task_state = self._grade_task(definition, effective_root)
        return EpisodeRewardState(
            task_id=task_id,
            runtime_root=str(effective_root),
            known_fact_ids=set(),
            last_health=task_state.health,
            done=task_state.done,
        )

    def evaluate_action(self, state: EpisodeRewardState, command: str) -> RewardComputation:
        definition = self.task_registry[state.task_id]
        runtime_root = Path(state.runtime_root)

        if state.done:
            task_state = self._grade_task(definition, runtime_root)
            signal = RewardSignal(
                health_delta=0.0,
                knowledge_delta=0.0,
                action_penalty=0.0,
                total_reward=0.0,
            )
            return RewardComputation(
                signal=signal,
                state=state,
                task_state=task_state,
                catastrophic=False,
            )

        task_state = self._grade_task(definition, runtime_root)
        catastrophic = self.is_catastrophic_action(command)

        if catastrophic:
            state.done = True
            signal = RewardSignal(
                health_delta=0.0,
                knowledge_delta=0.0,
                action_penalty=self.catastrophic_penalty,
                total_reward=self.catastrophic_penalty,
            )
            return RewardComputation(
                signal=signal,
                state=state,
                task_state=task_state,
                catastrophic=True,
            )

        knowledge_delta = self._knowledge_delta(definition, state, command)
        health_delta = task_state.health - state.last_health
        total_reward = health_delta + knowledge_delta + self.step_penalty

        state.last_health = task_state.health
        state.done = task_state.done

        signal = RewardSignal(
            health_delta=health_delta,
            knowledge_delta=knowledge_delta,
            action_penalty=self.step_penalty,
            total_reward=total_reward,
        )
        return RewardComputation(
            signal=signal,
            state=state,
            task_state=task_state,
            catastrophic=False,
        )

    def is_catastrophic_action(self, command: str) -> bool:
        return any(
            re.search(pattern, command, flags=re.IGNORECASE)
            for pattern in self.destructive_command_patterns
        )

    def _knowledge_delta(
        self,
        definition: TaskScenarioDefinition,
        state: EpisodeRewardState,
        command: str,
    ) -> float:
        task_module = get_task_module(state.task_id)
        reward = 0.0
        for trigger in definition.diagnostic_triggers:
            if trigger.fact_id in state.known_fact_ids:
                continue
            if task_module.command_reveals_fact(command, trigger):
                state.known_fact_ids.add(trigger.fact_id)
                reward += trigger.reward
        return reward

    def _grade_task(self, definition: TaskScenarioDefinition, runtime_root: Path) -> TaskScenarioState:
        task_module = get_task_module(definition.metadata.task_id)
        return task_module.grade(runtime_root)


def build_reward_engine(
    task_registry: dict[str, TaskScenarioDefinition],
    step_penalty: float = DEFAULT_STEP_PENALTY,
    catastrophic_penalty: float = DEFAULT_CATASTROPHIC_PENALTY,
    destructive_command_patterns: tuple[str, ...] = DEFAULT_DESTRUCTIVE_COMMAND_PATTERNS,
) -> RewardEngine:
    return RewardEngine(
        task_registry=task_registry,
        step_penalty=step_penalty,
        catastrophic_penalty=catastrophic_penalty,
        destructive_command_patterns=destructive_command_patterns,
    )


__all__ = [
    "DEFAULT_CATASTROPHIC_PENALTY",
    "DEFAULT_DESTRUCTIVE_COMMAND_PATTERNS",
    "DEFAULT_STEP_PENALTY",
    "EpisodeRewardState",
    "RewardComputation",
    "RewardEngine",
    "build_reward_engine",
]
