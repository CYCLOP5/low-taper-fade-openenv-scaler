from pathlib import Path

from sysadmin_env.tasks import TASK_MODULES
from sysadmin_env.tasks import build_task_registry
from sysadmin_env.tasks import disk_full
from sysadmin_env.tasks import network_broken
from sysadmin_env.tasks import nginx_crash


def test_task_registry_contains_phase_four_tasks():
    registry = build_task_registry("/assets")
    assert set(registry.keys()) == {"nginx_crash", "disk_full", "network_broken"}
    assert registry["nginx_crash"].metadata.base_filesystem_path == "/assets/nginx_crash"
    assert registry["disk_full"].metadata.max_steps > 0
    assert registry["network_broken"].metadata.time_limit > 0


def test_nginx_crash_prepare_and_grade(tmp_path: Path):
    nginx_crash.prepare_filesystem(tmp_path)
    initial = nginx_crash.grade(tmp_path)
    assert initial.health == 0.0
    assert not initial.done
    assert not initial.details["stale_pid_removed"]
    assert not initial.details["config_fixed"]

    (tmp_path / nginx_crash.PID_PATH).unlink()
    (tmp_path / nginx_crash.CONFIG_PATH).write_text(nginx_crash.FIXED_CONFIG)
    (tmp_path / nginx_crash.RUNNING_FLAG_PATH).write_text("running\n")

    fixed = nginx_crash.grade(tmp_path)
    assert fixed.health == nginx_crash.COMPLETION_HEALTH
    assert fixed.done
    assert fixed.details["service_running"]


def test_nginx_crash_triggers_match_expected_commands():
    triggers = {trigger.fact_id: trigger for trigger in nginx_crash.diagnostic_triggers()}
    assert nginx_crash.command_reveals_fact("cat /var/log/nginx/error.log", triggers["nginx_error_log_checked"])
    assert nginx_crash.command_reveals_fact("nginx -t", triggers["nginx_config_tested"])
    assert nginx_crash.command_reveals_fact("ps aux | grep nginx", triggers["nginx_process_table_checked"])


def test_disk_full_prepare_and_grade(tmp_path: Path):
    disk_full.prepare_filesystem(tmp_path)
    initial = disk_full.grade(tmp_path)
    assert initial.health == 0.0
    assert not initial.done

    (tmp_path / disk_full.DISCOVERY_PATH).write_text("full\n")
    partial = disk_full.grade(tmp_path)
    assert partial.details["filesystem_identified"]
    assert partial.health == 0.3

    (tmp_path / disk_full.HIDDEN_LOG_PATH).unlink()
    (tmp_path / disk_full.USAGE_PATH).write_text("10\n")
    complete = disk_full.grade(tmp_path)
    assert complete.health == disk_full.COMPLETION_HEALTH
    assert complete.done
    assert complete.details["filesystem_has_capacity"]


def test_disk_full_triggers_match_expected_commands():
    triggers = {trigger.fact_id: trigger for trigger in disk_full.diagnostic_triggers()}
    assert disk_full.command_reveals_fact("df -h", triggers["disk_usage_checked"])
    assert disk_full.command_reveals_fact("du -sh /mnt/data", triggers["large_files_checked"])
    assert disk_full.command_reveals_fact("find /mnt/data -type f", triggers["hidden_files_checked"])
    assert disk_full.command_reveals_fact("lsof", triggers["open_files_checked"])


def test_network_broken_prepare_and_grade(tmp_path: Path):
    network_broken.prepare_filesystem(tmp_path)
    initial = network_broken.grade(tmp_path)
    assert initial.health == 0.0
    assert not initial.done

    (tmp_path / network_broken.PING_FLAG_PATH).write_text("diagnosed\n")
    diagnosed = network_broken.grade(tmp_path)
    assert diagnosed.health == 0.2
    assert diagnosed.details["routing_issue_diagnosed"]

    (tmp_path / network_broken.ROUTE_PATH).write_text(network_broken.FIXED_ROUTE)
    (tmp_path / network_broken.RESOLV_PATH).write_text(network_broken.FIXED_RESOLV)
    fixed = network_broken.grade(tmp_path)
    assert fixed.health == network_broken.COMPLETION_HEALTH
    assert fixed.done
    assert fixed.details["outbound_connectivity_restored"]


def test_network_broken_triggers_match_expected_commands():
    triggers = {trigger.fact_id: trigger for trigger in network_broken.diagnostic_triggers()}
    assert network_broken.command_reveals_fact("ip route show", triggers["routes_checked"])
    assert network_broken.command_reveals_fact("ip addr", triggers["addresses_checked"])
    assert network_broken.command_reveals_fact("ip link", triggers["links_checked"])
    assert network_broken.command_reveals_fact("ping 1.1.1.1", triggers["connectivity_checked"])
    assert network_broken.command_reveals_fact("cat /etc/resolv.conf", triggers["dns_checked"])


def test_task_modules_export_expected_ids():
    assert set(TASK_MODULES.keys()) == {nginx_crash.TASK_ID, disk_full.TASK_ID, network_broken.TASK_ID}
