from __future__ import annotations

import json
import re
from pathlib import Path

from sysadmin_env.models import DiagnosticTrigger
from sysadmin_env.models import DifficultyTier
from sysadmin_env.models import TaskMetadata
from sysadmin_env.models import TaskScenarioDefinition
from sysadmin_env.models import TaskScenarioState
from sysadmin_env.tasks import hpc_outage


TASK_ID = "hpc_ood_apache"
COMPLETION_HEALTH = 1.0

SHARED_STATE_PATH = hpc_outage.SHARED_STATE_PATH
LOGIN_ROOT = hpc_outage.LOGIN_ROOT
HTTPD_CONF_RELATIVE = Path("etc/httpd/conf/httpd.conf")
HTTPD_CONF_PATH = LOGIN_ROOT / HTTPD_CONF_RELATIVE
APACHECTL_RELATIVE = Path("usr/local/bin/apachectl")

FIXED_HTTPD_CONF = (
    "ServerName hpc-login\n"
    "Listen 8081\n"
    "DocumentRoot /var/www/ood\n"
    "Include conf.modules.d/00-mpm.conf\n"
    "ErrorLog /var/log/httpd/error_log\n"
)

BROKEN_HTTPD_CONF = (
    "ServerName hpc-login\n"
    "Listn 8081\n"
    "DocumentRoot /var/www/ood\n"
    "Include conf.modules.d/00-mpm.conf\n"
    "ErrorLog /var/log/httpd/error_log\n"
)

INITIAL_STATE: dict = {
    "cluster": "rocky-hpc",
    "cores_total": hpc_outage.CLUSTER_CORES_TOTAL,
    "cores_per_node": hpc_outage.CLUSTER_CORES_PER_NODE,
    "partitions": {
        "compute": {"nodes": ["compute-01"], "default": True},
    },
    "nodes": {
        "login": {
            "state": "up",
            "reason": "",
            "cores": hpc_outage.CLUSTER_CORES_PER_NODE,
        },
        "compute-01": {
            "state": "idle",
            "reason": "",
            "cores": hpc_outage.CLUSTER_CORES_PER_NODE,
        },
    },
    "services": {
        "slurmd@login": "active",
        "slurmd@compute-01": "active",
        "slurmctld@login": "active",
        "httpd@login": "failed",
    },
    "portals": {
        "apache_ood": {
            "port": 8081,
            "state": "degraded",
            "last_error": "AH00526: Syntax error in file /etc/httpd/conf/httpd.conf: Listn",
        },
    },
    "jobs": [
        {
            "id": 12044,
            "name": "weather_ensemble",
            "user": "meteo",
            "state": "R",
            "partition": "compute",
            "nodes": "compute-01",
            "time": "1:04:21",
        },
    ],
}


def build_definition(base_filesystem_path: str) -> TaskScenarioDefinition:
    metadata = TaskMetadata(
        task_id=TASK_ID,
        difficulty=DifficultyTier.medium,
        description="open ondemand apache portal on :8081 returns 500 due to a one character typo in httpd.conf",
        max_steps=80,
        time_limit=540.0,
        base_filesystem_path=base_filesystem_path,
    )
    return TaskScenarioDefinition(
        metadata=metadata,
        requires_network_isolation=False,
        allows_nested_sandbox=True,
        diagnostic_triggers=diagnostic_triggers(),
    )


def diagnostic_triggers() -> list[DiagnosticTrigger]:
    return [
        DiagnosticTrigger(
            fact_id="cluster_queue_inspected",
            command_patterns=[r"\bsinfo\b", r"\bsqueue\b"],
            reward=0.04,
        ),
        DiagnosticTrigger(
            fact_id="httpd_service_checked",
            command_patterns=[r"systemctl\s+status\s+httpd", r"systemctl\s+is-failed\s+httpd"],
            reward=0.06,
        ),
        DiagnosticTrigger(
            fact_id="httpd_conf_inspected",
            command_patterns=[r"cat\s+.+httpd\.conf", r"grep\s+.+httpd\.conf"],
            reward=0.06,
        ),
        DiagnosticTrigger(
            fact_id="apachectl_configtest_run",
            command_patterns=[r"\bapachectl\s+configtest\b", r"\bhttpd\s+-t\b"],
            reward=0.08,
        ),
        DiagnosticTrigger(
            fact_id="apache_portal_probed",
            command_patterns=[r"curl\s+.+localhost:8081", r"curl\s+.+127\.0\.0\.1:8081"],
            reward=0.05,
        ),
    ]


def prepare_filesystem(root: str | Path) -> None:
    root_path = Path(root)
    hpc_outage.prepare_filesystem(root_path)

    route_path = root_path / hpc_outage.COMPUTE_ROUTE_PATH
    route_path.parent.mkdir(parents=True, exist_ok=True)
    route_path.write_text(hpc_outage.FIXED_ROUTE)

    conf_path = root_path / HTTPD_CONF_PATH
    conf_path.parent.mkdir(parents=True, exist_ok=True)
    conf_path.write_text(BROKEN_HTTPD_CONF)

    _write_executable(root_path / APACHECTL_RELATIVE, _apachectl_stub())
    login_bin = root_path / LOGIN_ROOT / "usr/local/bin"
    login_bin.mkdir(parents=True, exist_ok=True)
    _write_executable(login_bin / "apachectl", _apachectl_stub())

    _write_state(root_path / SHARED_STATE_PATH, INITIAL_STATE)


def inject_fault(root: str | Path) -> None:
    prepare_filesystem(root)


def observe_command(root: str | Path, command: str, _result) -> None:
    _ = Path(root)
    _ = command


def synchronize(root: str | Path) -> None:
    root_path = Path(root)
    if not (root_path / SHARED_STATE_PATH).exists():
        _write_state(root_path / SHARED_STATE_PATH, INITIAL_STATE)


def grade(root: str | Path) -> TaskScenarioState:
    root_path = Path(root)
    state_doc = _read_state(root_path / SHARED_STATE_PATH)
    conf_text = _safe_read(root_path / HTTPD_CONF_PATH)

    conf_fixed = conf_text == FIXED_HTTPD_CONF
    httpd_service = state_doc.get("services", {}).get("httpd@login", "")
    httpd_active = httpd_service == "active"
    portal_state = state_doc.get("portals", {}).get("apache_ood", {}).get("state", "")
    portal_healthy = portal_state == "healthy"

    # service state only counts if the conf is fixed so naive systemctl restart does not win
    effective_httpd = httpd_active and conf_fixed

    health = 0.0
    if conf_fixed:
        health += 0.35
    if effective_httpd:
        health += 0.35
    if conf_fixed and effective_httpd and portal_healthy:
        health = COMPLETION_HEALTH

    done = conf_fixed and effective_httpd and portal_healthy

    return TaskScenarioState(
        health=health,
        done=done,
        details={
            "httpd_conf_fixed": conf_fixed,
            "httpd_service_active": httpd_active,
            "apache_portal_healthy": portal_healthy,
            "expected_conf_first_lines": "ServerName hpc-login / Listen 8081",
        },
    )


def command_reveals_fact(command: str, trigger: DiagnosticTrigger) -> bool:
    return any(re.search(pattern, command, flags=re.IGNORECASE) for pattern in trigger.command_patterns)


def _safe_read(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text()


def _write_executable(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    path.chmod(0o755)


def _write_state(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")


def _read_state(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text() or "{}")
    except json.JSONDecodeError:
        return {}


def _apachectl_stub() -> str:
    return """#!/usr/bin/env python3
import fcntl
import json
import os
import sys

STATE_PATH = "/mnt/shared/slurm_state.json"
# The httpd.conf lives under the login-node subtree so it is visible from
# the login sandbox root as /nodes/login/etc/httpd/conf/httpd.conf.
CONF_PATH = "/nodes/login/etc/httpd/conf/httpd.conf"
FIXED_CONF = (
    "ServerName hpc-login\\n"
    "Listen 8081\\n"
    "DocumentRoot /var/www/ood\\n"
    "Include conf.modules.d/00-mpm.conf\\n"
    "ErrorLog /var/log/httpd/error_log\\n"
)
VALID_DIRECTIVES = {
    "ServerName", "Listen", "DocumentRoot", "Include", "ErrorLog",
    "ServerAdmin", "LoadModule", "User", "Group",
}

def read_conf():
    try:
        with open(CONF_PATH, "r", encoding="utf-8") as fh:
            return fh.read()
    except FileNotFoundError:
        return ""

def configtest():
    text = read_conf()
    if not text:
        print("AH00014: Configuration check failed: httpd.conf missing")
        return 1
    for idx, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        directive = stripped.split()[0]
        if directive not in VALID_DIRECTIVES:
            print(
                f"AH00526: Syntax error on line {idx} of {CONF_PATH}: "
                f"Invalid directive '{directive}'"
            )
            return 1
    if text == FIXED_CONF:
        print("Syntax OK")
        return 0
    print("Syntax OK")
    return 0

def mutate_state(mutator):
    with open(STATE_PATH, "r+", encoding="utf-8") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            raw = fh.read()
            doc = json.loads(raw or "{}")
            mutator(doc)
            fh.seek(0)
            fh.truncate()
            fh.write(json.dumps(doc, indent=2, sort_keys=True) + "\\n")
            fh.flush()
            os.fsync(fh.fileno())
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)

def graceful():
    rc = configtest()
    if rc != 0:
        sys.stderr.write("apachectl graceful: refusing to reload, config invalid\\n")
        return rc
    def apply(doc):
        services = doc.setdefault("services", {})
        services["httpd@login"] = "active"
        portals = doc.setdefault("portals", {})
        ood = portals.setdefault("apache_ood", {})
        ood["state"] = "healthy"
        ood["last_error"] = ""
    mutate_state(apply)
    print("apachectl graceful: httpd reloaded, apache_ood portal marked healthy")
    return 0

def status():
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as fh:
            fcntl.flock(fh.fileno(), fcntl.LOCK_SH)
            try:
                doc = json.loads(fh.read() or "{}")
            finally:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    except FileNotFoundError:
        doc = {}
    svc = doc.get("services", {}).get("httpd@login", "inactive")
    portal = doc.get("portals", {}).get("apache_ood", {})
    print(f"httpd service: {svc}")
    print(f"apache portal: state={portal.get('state','unknown')} port={portal.get('port','8081')}")
    return 0

def main(argv):
    if len(argv) < 2:
        print("usage: apachectl {configtest|graceful|status|restart}")
        return 1
    cmd = argv[1]
    if cmd == "configtest" or cmd == "-t":
        return configtest()
    if cmd in {"graceful", "restart", "reload", "start"}:
        return graceful()
    if cmd == "status":
        return status()
    if cmd == "stop":
        def apply(doc):
            services = doc.setdefault("services", {})
            services["httpd@login"] = "inactive"
            portals = doc.setdefault("portals", {})
            ood = portals.setdefault("apache_ood", {})
            ood["state"] = "down"
        mutate_state(apply)
        print("apachectl stop: httpd stopped")
        return 0
    print(f"apachectl: unknown subcommand {cmd}")
    return 1

if __name__ == "__main__":
    sys.exit(main(sys.argv))
"""
