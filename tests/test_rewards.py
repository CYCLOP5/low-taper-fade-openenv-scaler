from pathlib import Path

from sysadmin_env.rewards import DEFAULT_CATASTROPHIC_PENALTY
from sysadmin_env.rewards import DEFAULT_STEP_PENALTY
from sysadmin_env.rewards import build_reward_engine
from sysadmin_env.tasks import TASK_MODULES
from sysadmin_env.tasks import build_task_registry
from sysadmin_env.tasks import disk_full
from sysadmin_env.tasks import network_broken
from sysadmin_env.tasks import nginx_crash


def _build_prepared_registry(tmp_path: Path):
    registry = build_task_registry(str(tmp_path))
    for task_id, module in TASK_MODULES.items():
        module.prepare_filesystem(Path(registry[task_id].metadata.base_filesystem_path))
    return registry


def test_reward_engine_grants_monotonic_one_time_knowledge_rewards_for_diagnostic_commands(tmp_path: Path):
    registry = _build_prepared_registry(tmp_path)
    engine = build_reward_engine(registry)
    state = engine.start_episode(nginx_crash.TASK_ID)

    first = engine.evaluate_action(state, "cat /var/log/nginx/error.log")
    second = engine.evaluate_action(state, "nginx -t")
    duplicate = engine.evaluate_action(state, "cat /var/log/nginx/error.log")

    assert first.signal.health_delta == 0.0
    assert first.signal.knowledge_delta == 0.05
    assert first.signal.total_reward == 0.05 + DEFAULT_STEP_PENALTY

    assert second.signal.knowledge_delta == 0.08
    assert second.signal.total_reward == 0.08 + DEFAULT_STEP_PENALTY
    assert second.signal.knowledge_delta > first.signal.knowledge_delta

    assert duplicate.signal.knowledge_delta == 0.0
    assert duplicate.signal.total_reward == DEFAULT_STEP_PENALTY
    assert state.known_fact_ids == {"nginx_error_log_checked", "nginx_config_tested"}


def test_reward_engine_tracks_health_delta_during_remediation_and_combines_total_reward(tmp_path: Path):
    registry = _build_prepared_registry(tmp_path)
    engine = build_reward_engine(registry)
    state = engine.start_episode(nginx_crash.TASK_ID)
    root = Path(registry[nginx_crash.TASK_ID].metadata.base_filesystem_path)

    first = engine.evaluate_action(state, "cat /var/log/nginx/error.log")
    assert first.signal.health_delta == 0.0
    assert first.signal.knowledge_delta == 0.05
    assert first.signal.total_reward == first.signal.health_delta + first.signal.knowledge_delta + first.signal.action_penalty

    (root / nginx_crash.PID_PATH).unlink()
    remediation = engine.evaluate_action(state, "rm /var/run/nginx.pid")
    assert remediation.signal.health_delta == 0.25
    assert remediation.signal.knowledge_delta == 0.0
    assert remediation.signal.total_reward == 0.25 + DEFAULT_STEP_PENALTY

    (root / nginx_crash.CONFIG_PATH).write_text(nginx_crash.FIXED_CONFIG)
    (root / nginx_crash.RUNNING_FLAG_PATH).write_text("running\n")
    completion = engine.evaluate_action(state, "service nginx start")

    assert completion.signal.health_delta == 0.74
    assert completion.signal.knowledge_delta == 0.0
    assert completion.signal.total_reward == 0.74 + DEFAULT_STEP_PENALTY
    assert completion.task_state.health == nginx_crash.COMPLETION_HEALTH
    assert completion.task_state.done
    assert state.done


def test_reward_engine_unknown_command_only_applies_step_penalty(tmp_path: Path):
    registry = _build_prepared_registry(tmp_path)
    engine = build_reward_engine(registry)
    state = engine.start_episode(disk_full.TASK_ID)

    result = engine.evaluate_action(state, "echo nothing useful")

    assert result.signal.health_delta == 0.0
    assert result.signal.knowledge_delta == 0.0
    assert result.signal.action_penalty == DEFAULT_STEP_PENALTY
    assert result.signal.total_reward == DEFAULT_STEP_PENALTY
    assert not result.catastrophic
    assert state.known_fact_ids == set()


def test_reward_engine_deduplicates_previously_discovered_facts(tmp_path: Path):
    registry = _build_prepared_registry(tmp_path)
    engine = build_reward_engine(registry)
    state = engine.start_episode(disk_full.TASK_ID)

    first = engine.evaluate_action(state, "df -h")
    second = engine.evaluate_action(state, "df")

    assert first.signal.knowledge_delta == 0.06
    assert second.signal.knowledge_delta == 0.0
    assert state.known_fact_ids == {"disk_usage_checked"}


def test_reward_engine_operates_across_all_registered_tasks(tmp_path: Path):
    registry = _build_prepared_registry(tmp_path)
    engine = build_reward_engine(registry)

    expected_commands = {
        nginx_crash.TASK_ID: ("nginx -t", 0.08, "nginx_config_tested"),
        disk_full.TASK_ID: ("find /mnt/data -type f", 0.06, "hidden_files_checked"),
        network_broken.TASK_ID: ("ip route show", 0.07, "routes_checked"),
    }

    for task_id, (command, expected_reward, fact_id) in expected_commands.items():
        state = engine.start_episode(task_id)
        result = engine.evaluate_action(state, command)

        assert result.signal.health_delta == 0.0
        assert result.signal.knowledge_delta == expected_reward
        assert result.signal.total_reward == expected_reward + DEFAULT_STEP_PENALTY
        assert fact_id in state.known_fact_ids


def test_reward_engine_detects_catastrophic_actions_and_terminates_immediately(tmp_path: Path):
    registry = _build_prepared_registry(tmp_path)
    engine = build_reward_engine(registry)
    state = engine.start_episode(network_broken.TASK_ID)

    result = engine.evaluate_action(state, "rm -rf /")

    assert result.catastrophic
    assert result.signal.health_delta == 0.0
    assert result.signal.knowledge_delta == 0.0
    assert result.signal.action_penalty == DEFAULT_CATASTROPHIC_PENALTY
    assert result.signal.total_reward == DEFAULT_CATASTROPHIC_PENALTY
    assert state.done

    after_done = engine.evaluate_action(state, "ip route show")
    assert after_done.signal.total_reward == 0.0
    assert after_done.signal.action_penalty == 0.0
    assert after_done.signal.knowledge_delta == 0.0
    assert not after_done.catastrophic
