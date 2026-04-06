from enum import Enum
from typing import Optional

from pydantic import BaseModel
from pydantic import Field


class DifficultyTier(str, Enum):
    easy = "easy"
    medium = "medium"
    hard = "hard"


class Action(BaseModel):
    command: str = Field(min_length=1)
    reasoning: Optional[str] = None


class Observation(BaseModel):
    stdout: str
    stderr: str
    exit_code: int
    working_directory: str
    execution_time: float = Field(ge=0.0)
    reward: float
    done: bool
    step_number: int = Field(ge=0)
    max_steps: int = Field(gt=0)


class TaskMetadata(BaseModel):
    task_id: str = Field(min_length=1)
    difficulty: DifficultyTier
    description: str
    max_steps: int = Field(gt=0)
    time_limit: float = Field(gt=0.0)
    base_filesystem_path: str


class RewardSignal(BaseModel):
    health_delta: float
    knowledge_delta: float = Field(ge=0.0)
    action_penalty: float = Field(le=0.0)
    total_reward: float
