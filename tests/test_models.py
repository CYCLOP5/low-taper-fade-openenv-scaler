from pydantic import ValidationError

from sysadmin_env.models import Action
from sysadmin_env.models import DifficultyTier
from sysadmin_env.models import Observation
from sysadmin_env.models import RewardSignal
from sysadmin_env.models import TaskMetadata


def test_action_valid():
    a = Action(command="systemctl restart nginx")
    assert a.command == "systemctl restart nginx"
    assert a.reasoning is None
    j = a.model_dump_json()
    assert "systemctl restart nginx" in j
    print("action valid ok")


def test_action_with_reasoning():
    a = Action(command="journalctl -u nginx", reasoning="checking logs for crash cause")
    assert a.reasoning == "checking logs for crash cause"
    roundtrip = Action.model_validate_json(a.model_dump_json())
    assert roundtrip == a
    print("action with reasoning ok")


def test_action_empty_command_rejected():
    try:
        Action(command="")
        raise AssertionError("should have raised")
    except ValidationError:
        pass
    print("action empty command rejected ok")


def test_action_missing_command_rejected():
    try:
        Action.model_validate({})
        raise AssertionError("should have raised")
    except ValidationError:
        pass
    print("action missing command rejected ok")


def test_observation_valid():
    o = Observation(
        stdout="active (running)",
        stderr="",
        exit_code=0,
        working_directory="/root",
        execution_time=0.042,
        reward=0.15,
        done=False,
        step_number=3,
        max_steps=50,
    )
    j = o.model_dump_json()
    roundtrip = Observation.model_validate_json(j)
    assert roundtrip == o
    print("observation valid ok")


def test_observation_negative_execution_time_rejected():
    try:
        Observation(
            stdout="",
            stderr="",
            exit_code=0,
            working_directory="/",
            execution_time=-1.0,
            reward=0.0,
            done=False,
            step_number=0,
            max_steps=10,
        )
        raise AssertionError("should have raised")
    except ValidationError:
        pass
    print("observation negative execution time rejected ok")


def test_observation_zero_max_steps_rejected():
    try:
        Observation(
            stdout="",
            stderr="",
            exit_code=0,
            working_directory="/",
            execution_time=0.0,
            reward=0.0,
            done=False,
            step_number=0,
            max_steps=0,
        )
        raise AssertionError("should have raised")
    except ValidationError:
        pass
    print("observation zero max steps rejected ok")


def test_task_metadata_valid():
    t = TaskMetadata(
        task_id="nginx_crash",
        difficulty=DifficultyTier.easy,
        description="nginx crashed with stale pid and config syntax error",
        max_steps=50,
        time_limit=300.0,
        base_filesystem_path="/assets/nginx_crash",
    )
    j = t.model_dump_json()
    roundtrip = TaskMetadata.model_validate_json(j)
    assert roundtrip == t
    assert roundtrip.difficulty == DifficultyTier.easy
    print("task metadata valid ok")


def test_task_metadata_invalid_difficulty_rejected():
    try:
        TaskMetadata(
            task_id="test",
            difficulty="legendary",
            description="invalid tier",
            max_steps=10,
            time_limit=60.0,
            base_filesystem_path="/tmp/test",
        )
        raise AssertionError("should have raised")
    except ValidationError:
        pass
    print("task metadata invalid difficulty rejected ok")


def test_task_metadata_empty_id_rejected():
    try:
        TaskMetadata(
            task_id="",
            difficulty="easy",
            description="empty id",
            max_steps=10,
            time_limit=60.0,
            base_filesystem_path="/tmp/test",
        )
        raise AssertionError("should have raised")
    except ValidationError:
        pass
    print("task metadata empty id rejected ok")


def test_reward_signal_valid():
    r = RewardSignal(
        health_delta=0.25,
        knowledge_delta=0.1,
        action_penalty=-0.01,
        total_reward=0.34,
    )
    j = r.model_dump_json()
    roundtrip = RewardSignal.model_validate_json(j)
    assert roundtrip == r
    print("reward signal valid ok")


def test_reward_signal_positive_penalty_rejected():
    try:
        RewardSignal(
            health_delta=0.0,
            knowledge_delta=0.0,
            action_penalty=0.5,
            total_reward=0.5,
        )
        raise AssertionError("should have raised")
    except ValidationError:
        pass
    print("reward signal positive penalty rejected ok")


def test_reward_signal_negative_knowledge_rejected():
    try:
        RewardSignal(
            health_delta=0.0,
            knowledge_delta=-0.1,
            action_penalty=-0.01,
            total_reward=-0.11,
        )
        raise AssertionError("should have raised")
    except ValidationError:
        pass
    print("reward signal negative knowledge rejected ok")


def test_all_models_serialization_roundtrip():
    """confirms all four models survive json roundtrip"""
    action = Action(command="ls -la", reasoning="listing files")
    obs = Observation(
        stdout="total 0",
        stderr="",
        exit_code=0,
        working_directory="/root",
        execution_time=0.001,
        reward=-0.01,
        done=False,
        step_number=1,
        max_steps=50,
    )
    task = TaskMetadata(
        task_id="disk_full",
        difficulty=DifficultyTier.medium,
        description="hidden sparse log file filling loopback mount",
        max_steps=75,
        time_limit=600.0,
        base_filesystem_path="/assets/disk_full",
    )
    reward = RewardSignal(
        health_delta=0.0,
        knowledge_delta=0.05,
        action_penalty=-0.01,
        total_reward=0.04,
    )

    for model_instance in [action, obs, task, reward]:
        json_str = model_instance.model_dump_json()
        restored = type(model_instance).model_validate_json(json_str)
        assert restored == model_instance

    print("all models serialization roundtrip ok")


if __name__ == "__main__":
    test_action_valid()
    test_action_with_reasoning()
    test_action_empty_command_rejected()
    test_action_missing_command_rejected()
    test_observation_valid()
    test_observation_negative_execution_time_rejected()
    test_observation_zero_max_steps_rejected()
    test_task_metadata_valid()
    test_task_metadata_invalid_difficulty_rejected()
    test_task_metadata_empty_id_rejected()
    test_reward_signal_valid()
    test_reward_signal_positive_penalty_rejected()
    test_reward_signal_negative_knowledge_rejected()
    test_all_models_serialization_roundtrip()
    print("all phase one tests passed")
