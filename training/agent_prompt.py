from __future__ import annotations

import re
from typing import Iterable

BASH_BLOCK_RE = re.compile(r"<bash>(.*?)</bash>", re.DOTALL)
FINAL_ANSWER_RE = re.compile(r"<done\s*/?>", re.IGNORECASE)

SYSTEM_PROMPT = """you are an hpc cluster sre agent inside a deterministic rocky linux sandbox.

the site is rocky-hpc. there is a login node and one compute node compute-01. a shared
nfs volume at /mnt/shared holds slurm_state.json which the sinfo, squeue, systemctl and
scontrol stubs read under fcntl locks. open ondemand runs on http://localhost:8080 and
returns 502 while the fault is active and 200 once the fault clears.

to move between nodes use the ssh stub for example ssh compute-01 then run diagnostics
there. uname -n and hostname reflect the current node. the login node has no access to
/nodes directly you must ssh into the compute node to edit its files.

you act one shell command per turn. emit exactly one fenced action per reply using the
grammar below. nothing else in the reply is executed.

<bash>
single shell command here
</bash>

rules
- no multi line heredocs. if you need to write a file use printf or echo with > or tee
- no chained commands with && or ; keep a single atomic action
- you may use pipes within one command
- end the episode with <done/> after you have verified curl -I http://localhost:8080 is 200

typical remediation loop
1 sinfo and squeue to learn which compute node is drained
2 ssh compute-01 then inspect route-eth0, munge.key, and /var/run/slurmd.pid
3 repair the fault (fix the route file, chmod 0400 munge.key, or rm the stale pid)
4 systemctl restart munge and systemctl restart slurmd
5 exit back to login and curl -I http://localhost:8080 to verify 200
"""

USER_PROMPT = """incident: open ondemand returns 502 bad gateway and the compute partition is
drained. diagnose the root cause and restore service. respond with one bash action at a
time in the required grammar. keep actions short. do not narrate.
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
