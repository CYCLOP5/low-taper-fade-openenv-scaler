from __future__ import annotations

import json
import re
from pathlib import Path

from sysadmin_env.models import DiagnosticTrigger
from sysadmin_env.models import DifficultyTier
from sysadmin_env.models import TaskMetadata
from sysadmin_env.models import TaskScenarioDefinition
from sysadmin_env.models import TaskScenarioState


TASK_ID = "hpc_outage"
COMPLETION_HEALTH = 1.0

SHARED_STATE_PATH = Path("mnt/shared/slurm_state.json")
NODES_ROOT = Path("nodes")
LOGIN_ROOT = NODES_ROOT / "login"
COMPUTE_ROOT = NODES_ROOT / "compute-01"
ROUTE_RELATIVE = Path("etc/sysconfig/network-scripts/route-eth0")
COMPUTE_ROUTE_PATH = COMPUTE_ROOT / ROUTE_RELATIVE
OOD_DAEMON_PATH = Path("usr/local/bin/ood_server.py")

NODE_NAMES = ("login", "compute-01")
CLUSTER_CORES_TOTAL = 224
CLUSTER_CORES_PER_NODE = 112

FIXED_ROUTE = (
    "ADDRESS0=10.10.0.0\n"
    "NETMASK0=255.255.0.0\n"
    "GATEWAY0=10.10.1.1\n"
    "DEVICE0=eth0\n"
)

BROKEN_ROUTE = (
    "ADDRESS0=10.10.0.0\n"
    "NETMASK0=not_a_netmask\n"
    "GATEWAY0=0.0.0.0\n"
    "DEVICE0=eth9\n"
)

INITIAL_STATE: dict = {
    "cluster": "rocky-hpc",
    "cores_total": CLUSTER_CORES_TOTAL,
    "cores_per_node": CLUSTER_CORES_PER_NODE,
    "partitions": {
        "compute": {"nodes": ["compute-01"], "default": True},
    },
    "nodes": {
        "login": {
            "state": "up",
            "reason": "",
            "cores": CLUSTER_CORES_PER_NODE,
        },
        "compute-01": {
            "state": "drain",
            "reason": "slurmd failed route broken",
            "cores": CLUSTER_CORES_PER_NODE,
        },
    },
    "services": {
        "slurmd@login": "active",
        "slurmd@compute-01": "failed",
        "slurmctld@login": "active",
    },
    "jobs": [
        {
            "id": 4242,
            "name": "molecular_dynamics",
            "user": "researcher",
            "state": "PD",
            "partition": "compute",
            "nodes": "(Resources)",
            "time": "0:00",
        },
    ],
}


def build_definition(base_filesystem_path: str) -> TaskScenarioDefinition:
    metadata = TaskMetadata(
        task_id=TASK_ID,
        difficulty=DifficultyTier.hard,
        description="multi node hpc cluster outage with drained compute and broken ood portal",
        max_steps=90,
        time_limit=600.0,
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
            reward=0.06,
        ),
        DiagnosticTrigger(
            fact_id="compute_node_entered",
            command_patterns=[r"\bssh\s+compute-01\b"],
            reward=0.07,
        ),
        DiagnosticTrigger(
            fact_id="route_file_inspected",
            command_patterns=[r"cat\s+.+route-eth0", r"grep\s+.+route-eth0", r"ls\s+.+network-scripts"],
            reward=0.05,
        ),
        DiagnosticTrigger(
            fact_id="slurmd_service_checked",
            command_patterns=[r"systemctl\s+status\s+slurmd", r"systemctl\s+is-failed\s+slurmd"],
            reward=0.05,
        ),
        DiagnosticTrigger(
            fact_id="ood_portal_probed",
            command_patterns=[r"curl\s+.+localhost:8080", r"curl\s+.+127\.0\.0\.1:8080"],
            reward=0.05,
        ),
    ]


def prepare_filesystem(root: str | Path) -> None:
    root_path = Path(root)

    for relative in [
        Path("mnt/shared"),
        Path("usr/local/bin"),
        Path("etc"),
        Path("root"),
        Path("tmp"),
        Path("var/log/slurm"),
        Path("run"),
    ]:
        (root_path / relative).mkdir(parents=True, exist_ok=True)

    for node in NODE_NAMES:
        _build_node_skeleton(root_path / NODES_ROOT / node, node)

    (root_path / COMPUTE_ROUTE_PATH).parent.mkdir(parents=True, exist_ok=True)
    (root_path / COMPUTE_ROUTE_PATH).write_text(BROKEN_ROUTE)

    _write_state(root_path / SHARED_STATE_PATH, INITIAL_STATE)

    _write_executable(root_path / "usr/local/bin/ssh", _ssh_stub())
    _write_executable(root_path / "usr/local/bin/sinfo", _sinfo_stub())
    _write_executable(root_path / "usr/local/bin/squeue", _squeue_stub())
    _write_executable(root_path / "usr/local/bin/systemctl", _systemctl_stub())
    _write_executable(root_path / "usr/local/bin/scontrol", _scontrol_stub())
    _write_executable(root_path / "usr/local/bin/curl", _curl_stub())
    _write_executable(root_path / OOD_DAEMON_PATH, _ood_daemon_source())

    for node in NODE_NAMES:
        node_bin = root_path / NODES_ROOT / node / "usr/local/bin"
        node_bin.mkdir(parents=True, exist_ok=True)
        _write_executable(node_bin / "sinfo", _sinfo_stub())
        _write_executable(node_bin / "squeue", _squeue_stub())
        _write_executable(node_bin / "systemctl", _systemctl_stub())
        _write_executable(node_bin / "scontrol", _scontrol_stub())
        _write_executable(node_bin / "curl", _curl_stub())
        _write_executable(node_bin / "ssh", _ssh_stub())


def inject_fault(root: str | Path) -> None:
    prepare_filesystem(root)


def observe_command(root: str | Path, command: str, _result) -> None:
    _ = Path(root)
    _ = command


def synchronize(root: str | Path) -> None:
    root_path = Path(root)
    state_path = root_path / SHARED_STATE_PATH
    if not state_path.exists():
        _write_state(state_path, INITIAL_STATE)


def grade(root: str | Path) -> TaskScenarioState:
    root_path = Path(root)
    route_text = _safe_read(root_path / COMPUTE_ROUTE_PATH)
    route_fixed = route_text == FIXED_ROUTE

    state_doc = _read_state(root_path / SHARED_STATE_PATH)
    node_state = (
        state_doc.get("nodes", {})
        .get("compute-01", {})
        .get("state", "")
    )
    node_idle = node_state == "idle"

    health = 0.0
    if route_fixed:
        health += 0.3
    if node_idle:
        health += 0.3
    if route_fixed and node_idle:
        health = COMPLETION_HEALTH

    done = route_fixed and node_idle

    return TaskScenarioState(
        health=health,
        done=done,
        details={
            "route_file_restored": route_fixed,
            "compute_node_idle": node_idle,
            "compute_node_state": node_state or "unknown",
            "expected_route": FIXED_ROUTE,
        },
    )


def command_reveals_fact(command: str, trigger: DiagnosticTrigger) -> bool:
    return any(re.search(pattern, command, flags=re.IGNORECASE) for pattern in trigger.command_patterns)


def _build_node_skeleton(node_root: Path, hostname: str) -> None:
    for relative in [
        Path("etc"),
        Path("etc/sysconfig/network-scripts"),
        Path("usr/local/bin"),
        Path("var/log/slurm"),
        Path("root"),
        Path("tmp"),
        Path("run"),
        Path("home"),
    ]:
        (node_root / relative).mkdir(parents=True, exist_ok=True)
    (node_root / "etc/hostname").write_text(f"{hostname}\n")
    (node_root / "etc/motd").write_text(f"welcome to {hostname} rocky 9 hpc node\n")


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


def _safe_read(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text()


def _ssh_stub() -> str:
    return """#!/bin/bash
# simulated ssh that shifts root into /nodes/$TARGET via nested bwrap
TARGET=""
REMOTE_CMD=()
while [ $# -gt 0 ]; do
    case "$1" in
        -o|-i|-p|-l)
            shift 2
            ;;
        -*)
            shift
            ;;
        *)
            if [ -z "$TARGET" ]; then
                TARGET="$1"
            else
                REMOTE_CMD+=("$1")
            fi
            shift
            ;;
    esac
done

if [ -z "$TARGET" ]; then
    echo "ssh: missing destination host" >&2
    exit 1
fi

CLEAN_TARGET="${TARGET#*@}"
if [ ! -d "/nodes/$CLEAN_TARGET" ]; then
    echo "ssh: Could not resolve hostname $CLEAN_TARGET: Name or service not known" >&2
    exit 255
fi

BWRAP_BIN="$(command -v bwrap || echo /usr/bin/bwrap)"
if [ ! -x "$BWRAP_BIN" ]; then
    echo "ssh: nested sandbox runtime bwrap is unavailable" >&2
    exit 255
fi

BWRAP_ARGS=(
    --bind "/nodes/$CLEAN_TARGET" /
    --bind /mnt/shared /mnt/shared
    --ro-bind /usr/bin /usr/bin
    --ro-bind /bin /bin
    --proc /proc
    --dev /dev
    --tmpfs /tmp
    --unshare-pid
    --unshare-uts
    --hostname "$CLEAN_TARGET"
    --setenv HOSTNAME "$CLEAN_TARGET"
    --setenv PS1 "[root@$CLEAN_TARGET \\W]\\$ "
    --setenv PATH "/usr/local/bin:/usr/bin:/bin"
    --setenv TERM "${TERM:-xterm}"
    --chdir /root
)

for hostdir in /usr/sbin /sbin /lib /lib64 /usr/lib /usr/lib64 /usr/share /etc/alternatives /etc/ld.so.cache; do
    if [ -e "$hostdir" ]; then
        BWRAP_ARGS+=(--ro-bind "$hostdir" "$hostdir")
    fi
done

if [ ${#REMOTE_CMD[@]} -eq 0 ]; then
    exec "$BWRAP_BIN" "${BWRAP_ARGS[@]}" -- /bin/bash --login
fi

exec "$BWRAP_BIN" "${BWRAP_ARGS[@]}" -- /bin/bash -lc "${REMOTE_CMD[*]}"
"""


def _sinfo_stub() -> str:
    return """#!/usr/bin/env python3
import fcntl
import json
import os
import sys

STATE_PATH = "/mnt/shared/slurm_state.json"

def load_state():
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as fh:
            fcntl.flock(fh.fileno(), fcntl.LOCK_SH)
            try:
                data = fh.read()
            finally:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    except FileNotFoundError:
        sys.stderr.write("sinfo: slurm state file is missing\\n")
        sys.exit(1)
    try:
        return json.loads(data or "{}")
    except json.JSONDecodeError as exc:
        sys.stderr.write(f"sinfo: slurm state unreadable {exc}\\n")
        sys.exit(1)

def main():
    doc = load_state()
    partitions = doc.get("partitions", {})
    nodes = doc.get("nodes", {})
    header = f"{'PARTITION':<12}{'AVAIL':<8}{'TIMELIMIT':<12}{'NODES':<7}{'STATE':<10}NODELIST"
    print(header)
    for partition, pdef in partitions.items():
        for node_name in pdef.get("nodes", []):
            ninfo = nodes.get(node_name, {})
            state = ninfo.get("state", "unknown")
            avail = "up" if state != "down" else "down"
            row = f"{partition:<12}{avail:<8}{'infinite':<12}{'1':<7}{state:<10}{node_name}"
            print(row)
    for node_name, ninfo in nodes.items():
        if node_name in {n for p in partitions.values() for n in p.get("nodes", [])}:
            continue
        state = ninfo.get("state", "unknown")
        print(f"{'(none)':<12}{'n/a':<8}{'n/a':<12}{'1':<7}{state:<10}{node_name}")

if __name__ == "__main__":
    main()
"""


def _squeue_stub() -> str:
    return """#!/usr/bin/env python3
import fcntl
import json
import sys

STATE_PATH = "/mnt/shared/slurm_state.json"

def load_state():
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as fh:
            fcntl.flock(fh.fileno(), fcntl.LOCK_SH)
            try:
                data = fh.read()
            finally:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    except FileNotFoundError:
        sys.stderr.write("squeue: slurm state file is missing\\n")
        sys.exit(1)
    try:
        return json.loads(data or "{}")
    except json.JSONDecodeError as exc:
        sys.stderr.write(f"squeue: slurm state unreadable {exc}\\n")
        sys.exit(1)

def main():
    doc = load_state()
    jobs = doc.get("jobs", [])
    header = f"{'JOBID':<8}{'PARTITION':<12}{'NAME':<22}{'USER':<12}{'ST':<4}{'TIME':<10}{'NODES':<7}NODELIST(REASON)"
    print(header)
    if not jobs:
        return
    for job in jobs:
        row = (
            f"{str(job.get('id','')):<8}"
            f"{str(job.get('partition','')):<12}"
            f"{str(job.get('name',''))[:21]:<22}"
            f"{str(job.get('user','')):<12}"
            f"{str(job.get('state','')):<4}"
            f"{str(job.get('time','')):<10}"
            f"{'1':<7}"
            f"{job.get('nodes','(none)')}"
        )
        print(row)

if __name__ == "__main__":
    main()
"""


def _systemctl_stub() -> str:
    return """#!/usr/bin/env python3
import fcntl
import json
import os
import socket
import sys

STATE_PATH = "/mnt/shared/slurm_state.json"
FIXED_ROUTE = (
    "ADDRESS0=10.10.0.0\\n"
    "NETMASK0=255.255.0.0\\n"
    "GATEWAY0=10.10.1.1\\n"
    "DEVICE0=eth0\\n"
)
ROUTE_PATH_LOCAL = "/etc/sysconfig/network-scripts/route-eth0"
ROUTE_PATH_REMOTE = "/nodes/compute-01/etc/sysconfig/network-scripts/route-eth0"

def current_hostname():
    host = os.environ.get("HOSTNAME")
    if host:
        return host.strip()
    try:
        return socket.gethostname()
    except OSError:
        return ""

def route_is_fixed():
    for candidate in (ROUTE_PATH_LOCAL, ROUTE_PATH_REMOTE):
        try:
            with open(candidate, "r", encoding="utf-8") as fh:
                if fh.read() == FIXED_ROUTE:
                    return True
        except FileNotFoundError:
            continue
    return False

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
    return doc

def read_state():
    with open(STATE_PATH, "r", encoding="utf-8") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_SH)
        try:
            raw = fh.read()
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    return json.loads(raw or "{}")

def unit_key(unit, host):
    base = unit.split(".")[0]
    if "@" in base:
        return base
    return f"{base}@{host}" if host else base

def handle_status(unit, host):
    try:
        doc = read_state()
    except FileNotFoundError:
        sys.stderr.write("systemctl: slurm state file is missing\\n")
        return 3
    key = unit_key(unit, host)
    services = doc.get("services", {})
    status = services.get(key, "inactive")
    if status == "active":
        print(f"{unit} - loaded active (running)")
        return 0
    if status == "failed":
        print(f"{unit} - loaded failed (Result: exit-code)")
        return 3
    print(f"{unit} - loaded inactive (dead)")
    return 3

def handle_is_failed(unit, host):
    try:
        doc = read_state()
    except FileNotFoundError:
        print("unknown")
        return 1
    key = unit_key(unit, host)
    status = doc.get("services", {}).get(key, "inactive")
    print(status)
    return 0 if status == "failed" else 1

def handle_restart(unit, host):
    base = unit.split(".")[0].split("@")[0]
    if base != "slurmd":
        def noop(doc):
            services = doc.setdefault("services", {})
            services[unit_key(unit, host)] = "active"
        mutate_state(noop)
        print(f"{unit} restarted")
        return 0

    if host != "compute-01":
        def remote_restart(doc):
            services = doc.setdefault("services", {})
            services[f"slurmd@{host or 'unknown'}"] = "active"
        mutate_state(remote_restart)
        print(f"{unit} restarted on {host or 'unknown'}")
        return 0

    fixed = route_is_fixed()

    def apply(doc):
        services = doc.setdefault("services", {})
        nodes = doc.setdefault("nodes", {})
        compute = nodes.setdefault("compute-01", {})
        if fixed:
            services["slurmd@compute-01"] = "active"
            compute["state"] = "idle"
            compute["reason"] = ""
        else:
            services["slurmd@compute-01"] = "failed"
            compute["state"] = "drain"
            compute["reason"] = "slurmd failed route broken"

    mutate_state(apply)

    if fixed:
        print("slurmd restarted on compute-01 node returned to idle")
        return 0
    sys.stderr.write("slurmd failed to come up route-eth0 configuration invalid\\n")
    return 1

def main(argv):
    if len(argv) < 2:
        sys.stderr.write("systemctl: missing command\\n")
        return 1
    action = argv[1]
    rest = argv[2:]
    if action in {"daemon-reload", "list-units"}:
        print("ok")
        return 0
    if not rest:
        sys.stderr.write(f"systemctl: {action} requires a unit\\n")
        return 1
    unit = rest[0]
    host = current_hostname()
    if action == "status":
        return handle_status(unit, host)
    if action == "is-failed":
        return handle_is_failed(unit, host)
    if action in {"restart", "start"}:
        return handle_restart(unit, host)
    if action == "stop":
        def stop(doc):
            services = doc.setdefault("services", {})
            services[unit_key(unit, host)] = "inactive"
        mutate_state(stop)
        print(f"{unit} stopped")
        return 0
    sys.stderr.write(f"systemctl: unsupported action {action}\\n")
    return 1

if __name__ == "__main__":
    sys.exit(main(sys.argv))
"""


def _scontrol_stub() -> str:
    return """#!/usr/bin/env python3
import fcntl
import json
import os
import sys

STATE_PATH = "/mnt/shared/slurm_state.json"

def main(argv):
    if len(argv) < 2:
        sys.stderr.write("scontrol: missing subcommand\\n")
        return 1
    if argv[1] == "show" and len(argv) >= 4 and argv[2] == "node":
        with open(STATE_PATH, "r", encoding="utf-8") as fh:
            fcntl.flock(fh.fileno(), fcntl.LOCK_SH)
            try:
                doc = json.loads(fh.read() or "{}")
            finally:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        node = doc.get("nodes", {}).get(argv[3], {})
        if not node:
            sys.stderr.write(f"scontrol: unknown node {argv[3]}\\n")
            return 1
        print(f"NodeName={argv[3]} State={node.get('state','unknown')} Reason={node.get('reason','') or 'none'}")
        return 0
    if argv[1] == "update" and len(argv) >= 3:
        kvs = dict(item.split("=", 1) for item in argv[2:] if "=" in item)
        node_name = kvs.get("NodeName")
        state = kvs.get("State")
        if not node_name or not state:
            sys.stderr.write("scontrol: update requires NodeName and State\\n")
            return 1
        with open(STATE_PATH, "r+", encoding="utf-8") as fh:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            try:
                doc = json.loads(fh.read() or "{}")
                node = doc.setdefault("nodes", {}).setdefault(node_name, {})
                node["state"] = state.lower()
                if "Reason" in kvs:
                    node["reason"] = kvs["Reason"]
                fh.seek(0)
                fh.truncate()
                fh.write(json.dumps(doc, indent=2, sort_keys=True) + "\\n")
            finally:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        print(f"node {node_name} updated to {state}")
        return 0
    sys.stderr.write(f"scontrol: unsupported subcommand {argv[1]}\\n")
    return 1

if __name__ == "__main__":
    sys.exit(main(sys.argv))
"""


def _curl_stub() -> str:
    return """#!/bin/sh
# minimal curl that speaks http/1.0 against the in-sandbox ood daemon
METHOD="GET"
URL=""
HEADONLY=0
for arg in "$@"; do
    case "$arg" in
        -I|--head) HEADONLY=1 ;;
        -X) ;;
        GET|POST|PUT|DELETE) METHOD="$arg" ;;
        http://*|https://*) URL="$arg" ;;
    esac
done
if [ -z "$URL" ]; then
    echo "curl: missing url" >&2
    exit 2
fi
host_port="${URL#http://}"
host_port="${host_port#https://}"
path="/"
case "$host_port" in
    */*) path="/${host_port#*/}"; host_port="${host_port%%/*}" ;;
esac
host="${host_port%%:*}"
port="${host_port##*:}"
if [ "$host" = "$port" ]; then
    port=80
fi
if [ "$HEADONLY" = "1" ]; then
    METHOD="HEAD"
fi
request="$(printf '%s %s HTTP/1.0\\r\\nHost: %s\\r\\nConnection: close\\r\\n\\r\\n' "$METHOD" "$path" "$host")"
if command -v python3 >/dev/null 2>&1; then
    python3 - "$host" "$port" "$request" <<'PY'
import socket, sys
host, port, request = sys.argv[1], int(sys.argv[2]), sys.argv[3]
try:
    with socket.create_connection((host, port), timeout=5) as sock:
        sock.sendall(request.encode())
        chunks = []
        while True:
            chunk = sock.recv(4096)
            if not chunk:
                break
            chunks.append(chunk)
        sys.stdout.write(b"".join(chunks).decode("utf-8", "replace"))
except OSError as exc:
    sys.stderr.write(f"curl: connection failed {exc}\\n")
    sys.exit(7)
PY
    exit $?
fi
echo "curl: python runtime unavailable" >&2
exit 2
"""


def _ood_daemon_source() -> str:
    return """#!/usr/bin/env python3
# open ondemand simulator: serves 200 when compute-01 routing is healthy else 502
import http.server
import os
import socketserver
import sys
import threading

PORT = int(os.environ.get("OOD_PORT", "8080"))
ROUTE_PATHS = (
    "/nodes/compute-01/etc/sysconfig/network-scripts/route-eth0",
    "/etc/sysconfig/network-scripts/route-eth0",
)
FIXED_ROUTE = (
    "ADDRESS0=10.10.0.0\\n"
    "NETMASK0=255.255.0.0\\n"
    "GATEWAY0=10.10.1.1\\n"
    "DEVICE0=eth0\\n"
)

def route_is_fixed():
    for path in ROUTE_PATHS:
        try:
            with open(path, "r", encoding="utf-8") as fh:
                if fh.read() == FIXED_ROUTE:
                    return True
        except FileNotFoundError:
            continue
    return False

class Handler(http.server.BaseHTTPRequestHandler):
    def _respond(self):
        if route_is_fixed():
            body = b"ok compute backend reachable"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        body = b"bad gateway compute-01 unreachable"
        self.send_response(502)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._respond()

    def do_HEAD(self):
        self._respond()

    def log_message(self, format, *args):
        sys.stderr.write("ood %s - %s\\n" % (self.address_string(), format % args))

class ReusableServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    allow_reuse_address = True
    daemon_threads = True

def main():
    with ReusableServer(("0.0.0.0", PORT), Handler) as server:
        sys.stderr.write(f"ood listening on :{PORT}\\n")
        try:
            server.serve_forever(poll_interval=0.2)
        except KeyboardInterrupt:
            pass

if __name__ == "__main__":
    main()
"""
