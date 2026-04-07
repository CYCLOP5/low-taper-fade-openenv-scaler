from __future__ import annotations

import re
from pathlib import Path

from sysadmin_env.models import DiagnosticTrigger
from sysadmin_env.models import DifficultyTier
from sysadmin_env.models import TaskMetadata
from sysadmin_env.models import TaskScenarioDefinition
from sysadmin_env.models import TaskScenarioState


TASK_ID = "network_broken"
COMPLETION_HEALTH = 0.99
ROUTE_PATH = Path("etc/network/routes/default")
ADDR_PATH = Path("etc/network/interfaces/eth0.addr")
LINK_PATH = Path("etc/network/interfaces/eth0.link")
RESOLV_PATH = Path("etc/resolv.conf")
PING_FLAG_PATH = Path("run/network.ping")

BROKEN_ROUTE = "default via 192.0.2.1 dev eth9\n"
FIXED_ROUTE = "default via 10.0.2.2 dev eth0\n"
BROKEN_RESOLV = "nameserver 0.0.0.0\n"
FIXED_RESOLV = "nameserver 1.1.1.1\n"


def build_definition(base_filesystem_path: str) -> TaskScenarioDefinition:
    metadata = TaskMetadata(
        task_id=TASK_ID,
        difficulty=DifficultyTier.hard,
        description="broken network namespace with corrupted routing and dns",
        max_steps=70,
        time_limit=480.0,
        base_filesystem_path=base_filesystem_path,
    )
    return TaskScenarioDefinition(
        metadata=metadata,
        requires_network_isolation=True,
        diagnostic_triggers=diagnostic_triggers(),
    )


def diagnostic_triggers() -> list[DiagnosticTrigger]:
    return [
        DiagnosticTrigger(
            fact_id="routes_checked",
            command_patterns=[r"ip\s+route\s+show", r"route\s+-n"],
            reward=0.07,
        ),
        DiagnosticTrigger(
            fact_id="addresses_checked",
            command_patterns=[r"ip\s+addr", r"ifconfig\b"],
            reward=0.05,
        ),
        DiagnosticTrigger(
            fact_id="links_checked",
            command_patterns=[r"ip\s+link", r"ethtool\b"],
            reward=0.05,
        ),
        DiagnosticTrigger(
            fact_id="connectivity_checked",
            command_patterns=[r"ping\b", r"curl\b"],
            reward=0.06,
        ),
        DiagnosticTrigger(
            fact_id="dns_checked",
            command_patterns=[r"cat\s+.+resolv\.conf", r"grep\s+.+resolv\.conf"],
            reward=0.05,
        ),
    ]


def prepare_filesystem(root: str | Path) -> None:
    root_path = Path(root)
    for relative in [
        Path("etc/network/routes"),
        Path("etc/network/interfaces"),
        Path("run"),
        Path("etc"),
        Path("usr/local/bin"),
        Path("root"),
    ]:
        (root_path / relative).mkdir(parents=True, exist_ok=True)
    (root_path / ROUTE_PATH).write_text(BROKEN_ROUTE)
    (root_path / ADDR_PATH).write_text("10.0.2.15/24\n")
    (root_path / LINK_PATH).write_text("up\n")
    (root_path / RESOLV_PATH).write_text(BROKEN_RESOLV)
    (root_path / PING_FLAG_PATH).write_text("broken\n")
    _write_executable(root_path / "usr/local/bin/ip", _ip_stub())
    _write_executable(root_path / "usr/local/bin/route", _route_stub())
    _write_executable(root_path / "usr/local/bin/ping", _ping_stub())


def inject_fault(root: str | Path) -> None:
    prepare_filesystem(root)


def observe_command(root: str | Path, command: str, _result) -> None:
    root_path = Path(root)
    if re.search(r"\b(ip\s+route|ip\s+addr|ip\s+link|ping|resolv\.conf)\b", command, flags=re.IGNORECASE):
        (root_path / PING_FLAG_PATH).write_text("diagnosed\n")


def synchronize(root: str | Path) -> None:
    root_path = Path(root)
    if not (root_path / PING_FLAG_PATH).exists():
        (root_path / PING_FLAG_PATH).write_text("broken\n")


def grade(root: str | Path) -> TaskScenarioState:
    root_path = Path(root)
    route_fixed = (root_path / ROUTE_PATH).read_text() == FIXED_ROUTE
    dns_fixed = (root_path / RESOLV_PATH).read_text() == FIXED_RESOLV
    connectivity_restored = _connectivity_restored(root_path, route_fixed, dns_fixed)
    routing_diagnosed = route_fixed or (root_path / PING_FLAG_PATH).read_text().strip() == "diagnosed"

    health = 0.0
    if routing_diagnosed:
        health += 0.2
    if route_fixed:
        health += 0.3
    if dns_fixed:
        health += 0.2
    if connectivity_restored:
        health += 0.29

    if connectivity_restored:
        health = COMPLETION_HEALTH

    return TaskScenarioState(
        health=health,
        done=connectivity_restored,
        details={
            "routing_issue_diagnosed": routing_diagnosed,
            "default_route_restored": route_fixed,
            "dns_resolution_restored": dns_fixed,
            "outbound_connectivity_restored": connectivity_restored,
        },
    )


def command_reveals_fact(command: str, trigger: DiagnosticTrigger) -> bool:
    return any(re.search(pattern, command, flags=re.IGNORECASE) for pattern in trigger.command_patterns)


def _connectivity_restored(root_path: Path, route_fixed: bool, dns_fixed: bool) -> bool:
    if not route_fixed or not dns_fixed:
        return False
    return (root_path / LINK_PATH).read_text().strip() == "up"


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content)
    path.chmod(0o755)


def _ip_stub() -> str:
    return """#!/bin/sh
routefile="/etc/network/routes/default"
addrfile="/etc/network/interfaces/eth0.addr"
linkfile="/etc/network/interfaces/eth0.link"
if [ "$1" = "route" ] && [ -z "$2" ]; then
    cat "$routefile"
    exit 0
fi
if [ "$1" = "route" ] && [ "$2" = "show" ]; then
    cat "$routefile"
    exit 0
fi
if [ "$1" = "route" ] && [ "$2" = "add" ] && [ "$3" = "default" ] && [ "$4" = "via" ] && [ -n "$5" ]; then
    printf 'default via %s dev eth0\n' "$5" > "$routefile"
    exit 0
fi
if [ "$1" = "route" ] && [ "$2" = "del" ] && [ "$3" = "default" ]; then
    : > "$routefile"
    exit 0
fi
if [ "$1" = "addr" ]; then
    addr="$(cat "$addrfile")"
    printf '%s\n' "2: eth0    inet $addr"
    exit 0
fi
if [ "$1" = "link" ]; then
    state="$(cat "$linkfile")"
    printf '%s\n' "2: eth0: <BROADCAST,MULTICAST,UP> mtu 1500 state $state"
    exit 0
fi
printf '%s\n' "unsupported ip command" >&2
exit 1
"""


def _route_stub() -> str:
    return """#!/bin/sh
exec ip route show
"""


def _ping_stub() -> str:
    return """#!/bin/sh
target=""
for arg in "$@"; do
    case "$arg" in
        -*) ;;
        [0-9]) ;;
        *) target="$arg" ;;
    esac
done
route="$(cat /etc/network/routes/default 2>/dev/null)"
dns="$(cat /etc/resolv.conf 2>/dev/null)"
link="$(cat /etc/network/interfaces/eth0.link 2>/dev/null)"
if [ "$link" != "up" ]; then
    printf '%s\n' "link down" >&2
    exit 1
fi
case "$target" in
    *[!0-9.]* )
        if [ "$route" = "default via 10.0.2.2 dev eth0" ] && [ "$dns" = "nameserver 1.1.1.1" ]; then
            printf '%s\n' "1 packets transmitted 1 received"
            exit 0
        fi
        printf '%s\n' "temporary failure in name resolution" >&2
        exit 1
        ;;
    * )
        if [ "$route" = "default via 10.0.2.2 dev eth0" ]; then
            printf '%s\n' "1 packets transmitted 1 received"
            exit 0
        fi
        printf '%s\n' "network is unreachable" >&2
        exit 1
        ;;
esac
"""
