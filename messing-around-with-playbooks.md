# playbook change notes

this document records the recent baseline-agent adjustments made while tuning the hard task, `network_broken`.

## goal

the goal of these changes was not to make the hard task trivial. it was to keep the baseline reproducible while removing prompt-side answer leakage and making failure modes easier to debug.

## change sequence

### 1. task playbook added explicit hard-task repair targets

the first prompt-oriented change added task guidance for the model path in `inference.py`.

**result**

- this made the baseline too strong on `network_broken`
- with `gpt-5.4-nano`, the task collapsed into a 2-step solve:
  1. write `nameserver 1.1.1.1`
  2. write `default via 10.0.2.2 dev eth0`

**interpretation**

the model was no longer solving the task from runtime evidence alone. the prompt had become too close to answer leakage.

### 2. prompt leakage removed from the `network_broken` playbook

the next change removed the exact route and resolver targets from the prompt-side playbook while keeping generic task guidance.

**result**

- the task stopped being trivially solved from the prompt
- however, the agent started falling into a repeated `ping -c 1 example.com` loop after the guardrail activated

**interpretation**

the guardrail was using an attempt-indexed fallback, so once it reached the tail of the task plan it kept repeating connectivity checks instead of applying the next unresolved repair.

### 3. state-aware guardrail added for `network_broken`

the fallback path was changed so that after enough diagnosis, the guardrail chooses the next unresolved repair in a fixed order:

1. repair dns
2. repair route
3. validate connectivity

**result**

- this removed the infinite `ping` loop caused by the earlier attempt-indexed fallback
- but the guardrail still advanced too early in one failure case because it treated a bad multi-nameserver dns write as if dns had already been fixed

### 4. strict repair detection added

repair detection was then tightened so that:

- exact canonical repair commands are always accepted
- broader repair-shaped commands only count if they actually produced a positive repair observation
- read-only commands like `cat /etc/resolv.conf` no longer count as repair signals

**result**

- the latest local `gpt-5.4-nano` run solved `network_broken` in 7 steps rather than 2
- the task now requires route/link/dns inspection first, then the guardrail applies dns repair and route repair in order

## latest observed local run summary

| task | success | steps | score |
| --- | --- | ---: | ---: |
| `nginx_crash` | `true` | `6` | `1.0` |
| `disk_full` | `true` | `4` | `1.0` |
| `network_broken` | `true` | `7` | `1.0` |

## so what we leartn

the final baseline is stronger than a naive generic model loop, but cleaner than the earlier prompt-leaking version.

the environment remains deterministic and benchmark-oriented, while the baseline now:

- avoids leaking the exact hard-task answer through the prompt
- exposes concise stderr guardrail traces for debugging
- keeps a reproducible recovery path for the hard task

the remaining benchmark-quality question is not whether the baseline runs, but how much of the hard task should be discoverable from environment observations versus baseline heuristics. this repository currently chooses a middle ground: generic prompt guidance, deterministic task graders, and a bounded state-aware guardrail for the hardest task.
