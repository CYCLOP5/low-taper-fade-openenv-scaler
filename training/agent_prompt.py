from __future__ import annotations

import re
from typing import Iterable

BASH_BLOCK_RE = re.compile(r"<bash>(.*?)</bash>", re.DOTALL)
FINAL_ANSWER_RE = re.compile(r"<done\s*/?>", re.IGNORECASE)

SYSTEM_PROMPT = """you are an hpc cluster sre agent inside a deterministic rocky linux sandbox.

the site is rocky-hpc. there is a login node and one compute node compute-01. a shared
nfs volume at /mnt/shared holds slurm_state.json which the sinfo, squeue, systemctl and
scontrol stubs read under fcntl locks. open ondemand runs on http://localhost:8080 and
returns 502 while the fault is active and 200 once the fault clears. a secondary apache
portal runs on http://localhost:8081 for the open ondemand dashboard.

to move between nodes use the ssh stub for example ssh compute-01 then run diagnostics
there. uname -n and hostname reflect the current node. the login node has no access to
/nodes directly you must ssh into the compute node to edit its files.

tooling you have access to includes sinfo, squeue, scontrol, systemctl, ssh, curl,
nvidia-smi (compute-01 only), mount, umount, apachectl (login node), chmod, rm, cat,
grep, printf, and ls. the nvidia-smi stub supports -q -d ECC for ecc counters and
-r -i 0 to reset a gpu. apachectl supports configtest, graceful, restart, status.
mount and umount manipulate the /mnt/shared nfs bind.

you act one shell command per turn. emit exactly one fenced action per reply using the
grammar below. nothing else in the reply is executed.

<bash>
single shell command here
</bash>

rules
- no multi line heredocs. if you need to write a file use printf or echo with > or tee
- no chained commands with && or ; keep a single atomic action
- you may use pipes within one command
- end the episode with <done/> after you have verified the fault is fully cleared

typical remediation loops
1 route/munge/pid: sinfo, ssh compute-01, inspect route-eth0 or munge.key or /var/run/slurmd.pid,
  repair, systemctl restart munge, systemctl restart slurmd, exit, curl -I :8080
2 gpu ecc: sinfo, ssh compute-01, nvidia-smi -q -d ECC, nvidia-smi -r -i 0, exit, curl -I :8080
3 nfs stale: sinfo, ssh compute-01, mount, umount -l /mnt/shared, mount /mnt/shared,
  systemctl restart slurmd, exit, curl -I :8080
4 apache ood: systemctl status httpd, cat /etc/httpd/conf/httpd.conf, apachectl configtest,
  printf '...' > /etc/httpd/conf/httpd.conf, apachectl graceful, curl -I :8081
"""

USER_PROMPT = """incident: the cluster or its open ondemand portals are degraded. diagnose the
root cause and restore service. the fault may be a broken route file, a bad munge key
mode, a stale slurmd pid, a gpu ecc error, a stale nfs mount, or an apache httpd.conf
typo. respond with one bash action at a time in the required grammar. keep actions short.
do not narrate.
"""


def render_messages(observation: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{USER_PROMPT}\n\ncurrent observation:\n{observation}"},
    ]


def parse_action(completion: str) -> tuple[str | None, bool]:
    match = BASH_BLOCK_RE.search(completion)
    command = match.group(1).strip() if match else None
    done = bool(FINAL_ANSWER_RE.search(completion))
    return command, done


def iter_actions(completion: str) -> Iterable[str]:
    for match in BASH_BLOCK_RE.finditer(completion):
        text = match.group(1).strip()
        if text:
            yield text
