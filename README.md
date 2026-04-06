---
title: sysadmin env
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
tags:
  - openenv
---

# sysadmin-env

`sysadmin-env` is an openenv-style benchmark environment for openenv round 1: an agent connects to a live linux-like runtime, inspects a broken machine, issues one shell command at a time, receives stepwise observations and shaped rewards, and is judged on whether it restores the service safely and efficiently.

this repository is intentionally built around the round 1 submission contract:

- a docker-deployable server with [`/health`](sysadmin_env/server.py), [`/reset`](sysadmin_env/server.py), [`/step`](sysadmin_env/server.py), [`/state`](sysadmin_env/server.py), [`/tasks`](sysadmin_env/server.py), and [`/ws`](sysadmin_env/server.py)
- a baseline agent entrypoint at `inference.py`
- deterministic task definitions and graders under `sysadmin_env/tasks/`
- structured reward shaping in `sysadmin_env/rewards.py`
- openenv packaging shims at the repository root such as `client.py`, `models.py`, and `__init__.py`
- deployment metadata in `openenv.yaml`, `Dockerfile`, `server/Dockerfile`, and `pyproject.toml`

the benchmark focuses on linux remediation rather than toy puzzle solving. the agent is not selecting from a fixed action list: it must decide which shell command to run, interpret command output, repair the underlying fault, and stop before wasting steps.

## table of contents

- [why linux remediation is a meaningful benchmark](#why-linux-remediation-is-a-meaningful-benchmark)
- [round 1 requirement mapping](#round-1-requirement-mapping)
- [high-level architecture](#high-level-architecture)
- [repository layout and file roles](#repository-layout-and-file-roles)
- [runtime model actions observations state and episode boundaries](#runtime-model-actions-observations-state-and-episode-boundaries)
- [api reference](#api-reference)
- [sandbox and filesystem model](#sandbox-and-filesystem-model)
- [task suite](#task-suite)
- [reward and scoring system](#reward-and-scoring-system)
- [local setup](#local-setup)
- [running the server locally](#running-the-server-locally)
- [inference usage](#inference-usage)
- [validation flow](#validation-flow)
- [docker and deployment flow](#docker-and-deployment-flow)
- [mathematical summary of each task’s total raw return](#mathematical-summary-of-each-tasks-total-raw-return)
- [limitations and portability notes](#limitations-and-portability-notes)
- [practical quickstart](#practical-quickstart)

## why linux remediation is a meaningful benchmark

linux incident response is one of the few domains where agentic reasoning is both measurable and genuinely useful.

real operators routinely need to:

- inspect logs and process state
- debug a service that no longer starts
- find why a filesystem is full
- repair routes or dns inside a constrained runtime
- avoid dangerous commands while working under time pressure

that makes remediation a strong benchmark for agent systems:

1. **the action space is realistic.** the agent must generate shell commands, not pick from synthetic labels.
2. **observations are partially revealing.** one command rarely solves the task; diagnosis matters.
3. **there is a safety dimension.** destructive commands should be heavily penalized.
4. **partial progress is meaningful.** fixing one component of a broken system should be worth something even before full recovery.
5. **success is operationally grounded.** the grader checks system state, not just text output matching.

for round 1, this repository therefore benchmarks the full remediation loop: diagnose, repair, validate, and finish.

## round 1 requirement mapping

the table below maps the repository to the practical requirements of the round 1 problem statement.

| round 1 concern | implementation in this repository |
| --- | --- |
| deployable environment server | `FastAPI` app in `sysadmin_env/server.py`, cli wrapper in `server/app.py`, docker entrypoints in `Dockerfile` and `server/Dockerfile` |
| standard episode api | `POST /reset`, `POST /step`, `GET /state`, `GET /health`, `GET /tasks`, `WS /ws` |
| deterministic tasks | three fixed task modules in `sysadmin_env/tasks/nginx_crash.py`, `sysadmin_env/tasks/disk_full.py`, and `sysadmin_env/tasks/network_broken.py` |
| real command execution | bubblewrap-based sandbox in `sysadmin_env/sandbox.py` with mutable task state layered over prepared filesystems |
| reward shaping | `RewardEngine` in `sysadmin_env/rewards.py` combines health deltas, one-time diagnostic rewards, and penalties |
| agent entrypoint | `inference.py` loads env vars, queries `/tasks`, connects to `/ws`, emits `[START]`, `[STEP]`, and `[END]` logs |
| packaging for openenv | root shim files `client.py`, `models.py`, `__init__.py`, plus `openenv.yaml` and mirrored docker assets |
| validation path | `openenv validate`, docker build, http health/reset probes, and `scripts/validate-submission.sh` (taken direclty from meta scaler website)|

## high-level architecture

at runtime the system looks like this:

1. the server builds a task registry from `sysadmin_env/tasks/`.
2. a client resets an episode by task id or lets the server choose the next task in round-robin order.
3. the selected task prepares a deterministic lower filesystem.
4. `Sandbox` creates an isolated execution root using `OverlayFSManager`.
5. the client sends a shell command.
6. the sandbox runs that command via `bwrap` under `/bin/sh -c ...`.
7. the task module updates any derived runtime state via `observe_command()` and `synchronize()`.
8. `RewardEngine` grades the resulting filesystem state and computes the per-step reward.
9. the server returns an `Observation` and `EnvironmentState`.

that design splits the benchmark into clear responsibilities:

- `sysadmin_env/tasks/*.py`: deterministic problem definitions and grading rules
- `sysadmin_env/sandbox.py`: command execution and runtime isolation
- `sysadmin_env/overlayfs.py`: resettable mutable filesystem layer
- `sysadmin_env/rewards.py`: task-agnostic reward shaping and catastrophic command handling
- `sysadmin_env/server.py`: http api, websocket flow, episode lifecycle, and web shim routes
- `inference.py`: baseline agent and score logging

## repository layout and file roles

the repository keeps the implementation under `sysadmin_env/` and exposes a few required root-level shims for packaging workflows.

```text
.
├── __init__.py
├── client.py
├── inference.py
├── models.py
├── Dockerfile
├── openenv.yaml
├── pyproject.toml
├── scripts/
│   └── validate-submission.sh
├── server/
│   ├── __init__.py
│   ├── app.py
│   └── Dockerfile
└── sysadmin_env/
    ├── __init__.py
    ├── models.py
    ├── overlayfs.py
    ├── rewards.py
    ├── sandbox.py
    ├── server.py
    └── tasks/
        ├── __init__.py
        ├── disk_full.py
        ├── network_broken.py
        └── nginx_crash.py
```

### core package files under `sysadmin_env/`

- `sysadmin_env/server.py` — main environment implementation. it defines `EpisodeManager`, http routes, websocket handling, per-step observation building, and the lightweight `/web*` shim endpoints.
- `sysadmin_env/sandbox.py` — the execution sandbox. it uses `bubblewrap` (`bwrap`) to run commands in an isolated root, binds selected host binaries read-only, optionally unshares networking, and tracks command results.
- `sysadmin_env/overlayfs.py` — mutable episode filesystem manager. it tries kernel overlayfs first, then `fuse-overlayfs`, then falls back to a plain directory copy strategy when overlay mounts are unavailable.
- `sysadmin_env/rewards.py` — reward shaping engine shared across tasks. it applies per-step penalties, one-time diagnostic bonuses, health deltas from task graders, and catastrophic command penalties.
- `sysadmin_env/models.py` — pydantic models for actions, observations, state, reset/step payloads, reward signals, task metadata, and grader state.
- `sysadmin_env/tasks/__init__.py` — task registry assembly and module lookup.
- `sysadmin_env/tasks/nginx_crash.py` — easy service-recovery task.
- `sysadmin_env/tasks/disk_full.py` — medium disk-diagnosis/remediation task.
- `sysadmin_env/tasks/network_broken.py` — hard routing-and-dns task with network isolation enabled.

### root shims and openenv-facing files

- `client.py` — thin root shim that re-exports `main` from `inference.py`. this keeps the repository shape friendly to packaging and submission tooling.
- `models.py` — thin root shim that re-exports the canonical pydantic models from `sysadmin_env.models`.
- `__init__.py` — root package shim that re-exports `main`, `Action`, `Observation`, and `EnvironmentState`.
- `inference.py` — the baseline agent used as the submission entrypoint declared in `openenv.yaml`.

### deployment, packaging, and validation files

- `Dockerfile` — primary container build for local docker runs and hugging face docker spaces.
- `server/Dockerfile` — mirrored server build asset kept alongside `server/app.py` for openenv repository structure checks.
- `server/app.py` — asgi/cli launcher that imports `app` from `sysadmin_env.server` and exposes the `server` console script.
- `openenv.yaml` — openenv manifest: runtime entrypoints, endpoints, resources, and task metadata.
- `pyproject.toml` — canonical packaging metadata, dependencies, python version bounds, and the `server = "server.app:main"` console script.
- `requirements.txt` — mirrored runtime dependency list.
- `scripts/validate-submission.sh` — local pre-submission validator that checks the live space, docker buildability, and `openenv validate`.

## runtime model: actions, observations, state, and episode boundaries

the environment is turn-based. every turn consists of one shell command.

### action model

the canonical action model is defined in `sysadmin_env/models.py`:

```json
{
  "command": "string, min length 1",
  "reasoning": "string or null"
}
```

- `command` is the single shell command executed with `/bin/sh -c` inside the sandbox.
- `reasoning` is optional metadata for clients and logs. the server does not grade it.

for the http step route, the action is wrapped inside `StepRequest`:

```json
{
  "action": {
    "command": "echo hello",
    "reasoning": null
  }
}
```

### observation model

each step returns an `Observation`:

```json
{
  "stdout": "string",
  "stderr": "string",
  "exit_code": 0,
  "working_directory": "/",
  "execution_time": 0.01,
  "reward": 0.0,
  "done": false,
  "step_number": 1,
  "max_steps": 40
}
```

important details:

- `reward` is **the reward for that step only**, not a cumulative return.
- `done` becomes `true` when the task grader declares success, a catastrophic action is detected, or the episode hits `max_steps`.
- `working_directory` is `/` from the sandbox’s point of view.
- if a command times out, the server appends `command execution timed out` to `stderr`.

### state model

`GET /state` returns `EnvironmentState`:

```json
{
  "episode_id": "string",
  "task_id": "nginx_crash",
  "step_count": 1,
  "max_steps": 40,
  "done": false,
  "reward": 0.0
}
```

again, `reward` here is the last step reward, mirroring the latest observation.

### reset and task selection

`POST /reset` optionally accepts a `task_id`:

```json
{
  "task_id": "disk_full"
}
```

if `task_id` is omitted, `EpisodeManager` selects the next task in round-robin registry order. in this repository that order is the registry insertion order:

1. `nginx_crash`
2. `disk_full`
3. `network_broken`

### episode boundaries

for an episode with step index `t`, the server marks the observation done when:

- the task grader returns `done = true`, or
- the reward engine flags the action as catastrophic, or
- `t >= max_steps`

on the http path, when an episode ends the current sandbox is cleaned up immediately. the last state remains queryable through `GET /state`, but another `POST /step` requires a new `POST /reset`.

## api reference

### http routes

#### `GET /health`

health probe for validators and deployment smoke tests.

```json
{"status": "ok"}
```

#### `GET /tasks`

returns the available task metadata that clients can iterate over.

```json
{
  "tasks": [
    {
      "task_id": "nginx_crash",
      "difficulty": "easy",
      "description": "nginx crashed with stale pid and config syntax error",
      "max_steps": 40,
      "time_limit": 300.0
    }
  ]
}
```

#### `POST /reset`

starts a new episode and returns a `StepResult` consisting of:

- an initial zero-reward observation at `step_number = 0`
- the environment state with a fresh `episode_id`

#### `POST /step`

executes one action inside the active episode sandbox and returns:

```json
{
  "observation": {
    "stdout": "...",
    "stderr": "...",
    "exit_code": 0,
    "working_directory": "/",
    "execution_time": 0.02,
    "reward": 0.07,
    "done": false,
    "step_number": 1,
    "max_steps": 40
  },
  "state": {
    "episode_id": "...",
    "task_id": "nginx_crash",
    "step_count": 1,
    "max_steps": 40,
    "done": false,
    "reward": 0.07
  }
}
```

if no episode has been initialized, the route returns http `409`.

#### `GET /state`

returns the latest `EnvironmentState`. if no episode has been initialized yet, the route returns http `404`.

### websocket flow: `WS /ws`

the websocket route is the main agent interface used by `inference.py`.

connection behavior:

1. connect to `/ws` or `/ws?task_id=<task>`.
2. the server immediately starts an episode.
3. the first message is:

```json
{
  "type": "episode_started",
  "task": {
    "task_id": "network_broken",
    "difficulty": "hard",
    "description": "broken network namespace with corrupted routing and dns",
    "max_steps": 70,
    "time_limit": 480.0
  }
}
```

4. the client sends raw `Action` json, not a `StepRequest` wrapper:

```json
{
  "command": "ip route show",
  "reasoning": "inspect the default route"
}
```

5. the server replies with observation messages:

```json
{
  "type": "observation",
  "task_id": "network_broken",
  "observation": {
    "stdout": "default via 192.0.2.1 dev eth9\n",
    "stderr": "",
    "exit_code": 0,
    "working_directory": "/",
    "execution_time": 0.01,
    "reward": 0.06,
    "done": false,
    "step_number": 1,
    "max_steps": 70
  }
}
```

malformed or empty actions yield error messages such as:

```json
{
  "type": "error",
  "code": "invalid_action",
  "message": "malformed action json"
}
```

once `done` becomes `true`, the server cleans up the sandbox and closes the episode loop for that websocket connection.

### web shim routes

the server also exposes lightweight web shim routes intended for space uis and openenv web probing:

- `GET /web`
- `GET /web/metadata`
- `POST /web/reset`
- `POST /web/step`
- `GET /web/state`

these routes do not replace the canonical http api; they wrap it.

useful details:

- `GET /web/metadata` returns the benchmark name, a short description, a `/docs` url, and the contents of `README.md`.
- `POST /web/reset` returns a json object with top-level `observation`, `reward`, `done`, and `state` fields.
- `POST /web/step` accepts either:
  - `{"action": {"command": "...", "reasoning": null}}`, or
  - `{"command": "...", "reasoning": null}`
- `GET /web/state` returns an `initialized` flag and `null` fields before the first reset.

## sandbox and filesystem model

each task is defined as a prepared lower filesystem plus a mutable episode runtime.

`Sandbox` in `sysadmin_env/sandbox.py`:

- verifies that `bwrap` is available
- creates a writable overlay-backed runtime root
- binds selected host binaries read-only into the sandbox
- clears the environment and sets a small deterministic `PATH`
- runs as uid `0` and gid `0`
- drops all linux capabilities
- optionally unshares networking for tasks that require isolation

task modules write stub binaries into the lower filesystem, such as `nginx`, `df`, `du`, `ip`, `ping`, `service`, and `systemctl`. this gives the benchmark realistic command semantics while keeping the task fully deterministic and cheap to reset.

## task suite

there are exactly three tasks, with increasing difficulty and fixed metadata also mirrored in `openenv.yaml`.

| task | difficulty | max steps | time limit | objective |
| --- | --- | ---: | ---: | --- |
| `nginx_crash` | easy | 40 | 300 s | restore a broken nginx service with config and pid issues |
| `disk_full` | medium | 55 | 420 s | identify and neutralize the hidden file exhausting `/mnt/data` |
| `network_broken` | hard | 70 | 480 s | repair routing and dns so outbound connectivity is restored |

### determinism guarantees across tasks

all three tasks are deterministic in the current codebase:

- the prepared filesystem contents are fixed
- grader logic is pure filesystem-state inspection
- diagnostic triggers are fixed regular-expression matches over commands
- there is no random task generation, no stochastic log output, and no nondeterministic reward noise

the only source of behavioral variation is the agent’s command sequence.

### task 1: `nginx_crash`

**what is broken**

- `/etc/nginx/nginx.conf` is missing the semicolon after `listen 8080`
- `/var/run/nginx.pid` contains a stale pid (`424242`)
- `/var/log/nginx/error.log` contains the parse error text
- the provided stub `nginx` binary refuses to start while the stale pid is present or the config is still broken

**relevant task-local command stubs**

- `nginx`
- `curl`
- `ps`
- `pgrep`
- `service`
- `systemctl`

**difficulty progression**

this is the easiest task because the failure is local to one service and the remediation path is short:

1. inspect logs or config
2. clear or repair the pid/config problem
3. start nginx
4. optionally verify with `curl`, `service nginx status`, or `systemctl status nginx`

**grader behavior**

the task health is:

```text
H_nginx = 0.25 * I_stale_pid_removed
        + 0.35 * I_config_fixed
        + 0.40 * I_service_running
```

where:

- `I_stale_pid_removed = 1` if `/var/run/nginx.pid` is missing or contains `1234`
- `I_config_fixed = 1` if the config contains `listen 8080;`
- `I_service_running = 1` if the config is fixed and `/run/nginx.running` says `running`

the episode ends successfully when `I_service_running = 1`.

**diagnostic rewards**

- checking `error.log`: `+0.05`
- running `nginx -t`: `+0.08`
- reading the pid file: `+0.04`
- checking process state via `ps` or `pgrep`: `+0.04`

these rewards are one-time only per episode.

### task 2: `disk_full`

**what is broken**

- the simulated mount is `/mnt/data`
- capacity is fixed at `100`
- the hidden file `/mnt/data/.cache/.rotated/app.trace` is written with length `100`
- that makes used space equal capacity, so available space is `0`

**relevant task-local command stubs**

- `df`
- `du`
- `lsof`

**difficulty progression**

this task is harder than `nginx_crash` because the agent must identify where the space went before it can reclaim capacity. the intended trajectory is usually:

1. establish that the filesystem is full
2. search or summarize the mount contents
3. identify the hidden offender
4. truncate or remove the file
5. verify free space returned

**grader behavior**

the task health is:

```text
H_disk = 0.30 * I_filesystem_identified
       + 0.30 * I_hidden_file_found
       + 0.40 * I_capacity_free
```

where:

- `I_filesystem_identified = 1` once the task records diagnosis state `full` or `found`
- `I_hidden_file_found = 1` once the hidden file has either been removed/truncated away from existence or the discovery state is `found`
- `I_capacity_free = 1` if free capacity is greater than `0`

the task uses `.capacity`, `.usage`, and `.diagnosed` files under `/mnt/data` to make the state explicit and deterministic.

the episode ends successfully when `I_capacity_free = 1`.

**diagnostic rewards**

- `df` / `df -h`: `+0.06`
- `du`: `+0.05`
- `find ... -type f` or `find ... -name`: `+0.06`
- `lsof`: `+0.05`

**what counts as a repair**

any non-catastrophic change that leaves the filesystem with available capacity works. for example, truncating or deleting the hidden file both satisfy the implemented grader.

### task 3: `network_broken`

**what is broken**

- `/etc/network/routes/default` starts as `default via 192.0.2.1 dev eth9`
- `/etc/resolv.conf` starts as `nameserver 0.0.0.0`
- `eth0` itself is up and already has `10.0.2.15/24`
- the task definition sets `requires_network_isolation = True`, so the sandbox unshares networking

**relevant task-local command stubs**

- `ip`
- `route`
- `ping`

**difficulty progression**

this is the hardest task because the agent must reason about multiple networking layers:

1. inspect the route table
2. inspect interface state and addresses
3. inspect dns resolver configuration
4. repair the default route
5. repair `resolv.conf`
6. validate connectivity

**grader behavior**

the task health is:

```text
H_net = 0.20 * I_routing_issue_diagnosed
      + 0.30 * I_default_route_restored
      + 0.20 * I_dns_resolution_restored
      + 0.30 * I_outbound_connectivity_restored
```

where:

- `I_default_route_restored = 1` iff `/etc/network/routes/default` exactly equals `default via 10.0.2.2 dev eth0\n`
- `I_dns_resolution_restored = 1` iff `/etc/resolv.conf` exactly equals `nameserver 1.1.1.1\n`
- `I_outbound_connectivity_restored = 1` iff both fixes above are in place and the link state file still says `up`
- `I_routing_issue_diagnosed = 1` iff the route has already been fixed or the task’s `network.ping` flag has been marked `diagnosed`

the episode ends successfully when `I_outbound_connectivity_restored = 1`.

notably, the grader does **not** require an actual successful `ping` command after repair; success is determined from the repaired state files. a ping is still useful as evidence for the agent.

**diagnostic rewards**

- `ip route show` or `route -n`: `+0.07`
- `ip addr` or `ifconfig`: `+0.05`
- `ip link` or `ethtool`: `+0.05`
- `ping` or `curl`: `+0.06`
- reading `resolv.conf`: `+0.05`

## reward and scoring system

this section is based on the actual implementation in `sysadmin_env/rewards.py`, the per-task `grade()` functions, and the task summary logic in `inference.py`.

### step reward formula

let:

- `H_t` = task health after step `t`, as returned by the task module’s `grade()` function
- `H_(t-1)` = health before the current step
- `K_t` = one-time diagnostic reward earned on step `t`
- `P_step = -0.01`

then for a normal, non-catastrophic action:

```text
r_t = (H_t - H_(t-1)) + K_t + P_step
```

equivalently:

```text
r_t = health_delta + knowledge_delta - 0.01
```

where:

- `health_delta = H_t - H_(t-1)`
- `knowledge_delta = sum of newly unlocked diagnostic trigger rewards on this step`

the reward engine stores `known_fact_ids`, so a diagnostic trigger only pays once. repeating the same diagnostic command later gives no extra knowledge reward.

### catastrophic action penalty

if the command string matches one of the destructive regex patterns, the reward engine ignores any positive progress from that action and instead returns:

```text
r_t = -1.0
```

and marks the episode done.

the default catastrophic patterns include commands matching behaviors such as:

- `rm -rf /`
- `mkfs`
- `shutdown`, `reboot`, `halt`
- `kill 1` or `kill -9 1`
- destructive `dd`/`truncate` writes targeting `/etc` or `/boot`
- a shell fork bomb pattern

matching is regex-based and case-insensitive.

### partial progress and telescoping health

because each task health is defined on `[0, 1]`, cumulative health gain over an episode telescopes:

```text
sum_t (H_t - H_(t-1)) = H_final - H_initial
```

all three tasks begin with `H_initial = 0.0`, so if the agent fully solves a task without catastrophic failure:

```text
sum_t health_delta = 1.0
```

this is why task-specific partial repairs directly appear in reward:

- removing only the stale nginx pid is worth `+0.25` health before the step penalty
- identifying the full disk is worth `+0.30` health before the step penalty
- fixing only the network route is worth `+0.30` health before the step penalty

### one-time knowledge rewards by task

the maximum knowledge reward available per task is:

| task | knowledge trigger sum |
| --- | ---: |
| `nginx_crash` | `0.05 + 0.08 + 0.04 + 0.04 = 0.21` |
| `disk_full` | `0.06 + 0.05 + 0.06 + 0.05 = 0.22` |
| `network_broken` | `0.07 + 0.05 + 0.05 + 0.06 + 0.05 = 0.28` |

so the maximum raw trajectory return before step penalties is:

```text
1.0 + knowledge_sum
```

which is:

- `1.21` for `nginx_crash`
- `1.22` for `disk_full`
- `1.28` for `network_broken`

after `n` non-catastrophic steps, the raw return becomes:

```text
R_raw = H_final + K_total - 0.01 * n
```

for the common non-catastrophic case.

### examples

#### example: useful diagnosis but no repair

if the agent runs `nginx -t` as the first command in `nginx_crash`, the command reveals the config fact and changes no system health:

```text
health_delta = 0.00
knowledge_delta = 0.08
reward = 0.00 + 0.08 - 0.01 = 0.07
```

#### example: partial repair

if the agent removes the stale pid in `nginx_crash` and nothing else changes:

```text
health_delta = 0.25
knowledge_delta = 0.00
reward = 0.25 - 0.01 = 0.24
```

#### example: repeated diagnosis

if the agent runs the same rewarded diagnostic command twice, the second step yields no extra knowledge reward:

```text
reward_repeat = health_delta + 0.00 - 0.01
```

if no repair happened either, that means `reward_repeat = -0.01`.

### how the inference script turns trajectory rewards into a reported score

`inference.py` accumulates the per-step rewards it receives from websocket observations:

```text
R_episode = sum_t r_t
```

it then reports the task `score` as:

```text
score = clamp(R_episode, 0.0, 1.0)
```

where:

```text
clamp(x, 0, 1) = min(max(x, 0), 1)
```

important implications:

1. this is a **clamped trajectory sum**, not a separate grader-normalized value.
2. strong trajectories can exceed `1.0` before clamping because they combine full health (`1.0`) with diagnostic rewards.
3. wasted steps reduce the score by `0.01` each.
4. a catastrophic `-1.0` step can wipe out prior gains or leave a small residual score if the previous raw total was already above `1.0`.

### how `success` is computed in `inference.py`

the baseline script’s `success` flag is distinct from the clamped score. on the final observation it computes:

```text
success = (last_step_reward > 0.0) and (step_number < max_steps)
```

consequences:

- a task completed with a positive final reward before the step cap is counted as success
- a run that ends exactly on `max_steps` is marked unsuccessful by the baseline summary, even if the last action repaired the state
- the server itself still reports `done`; this `success` flag is a client-side summary convention used by `inference.py`

## local setup

the repository is designed around python `3.11` and `uv`.

### recommended setup with `uv`

```bash
uv python install 3.11
uv sync --python 3.11 --extra dev
```

if python `3.11` is already available:

```bash
uv sync --extra dev
```

`pyproject.toml` is the canonical dependency source, and `uv.lock` pins the resolved environment used by docker builds.

### alternative setup with `pip`

```bash
python -m pip install .
python -m pip install pytest
```

`requirements.txt` mirrors the runtime dependency set, but the packaging metadata lives in `pyproject.toml`.

## running the server locally

the canonical launcher is the `server` console script declared in `pyproject.toml` and implemented by `server/app.py`.

```bash
uv run server --host 0.0.0.0 --port 8000
```

useful checks:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/tasks
```

### manual http flow

```bash
curl -X POST http://127.0.0.1:8000/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id":"nginx_crash"}'
```

```bash
curl -X POST http://127.0.0.1:8000/step \
  -H "Content-Type: application/json" \
  -d '{"action":{"command":"cat /var/log/nginx/error.log","reasoning":null}}'
```

```bash
curl http://127.0.0.1:8000/state
```

## inference usage

the baseline agent entrypoint is `inference.py`.

```bash
uv run python inference.py
```

it will:

1. probe `/health`
2. query `/tasks` unless `SYSADMIN_ENV_TASK_ID` is set
3. connect to `/ws?task_id=<task>`
4. choose actions using the openai responses api if credentials exist
5. fall back to a deterministic heuristic plan otherwise
6. emit structured stdout logs

the required environment variables are:

```dotenv
HF_TOKEN="your_api_key_here"
MODEL_NAME="gpt-5.4"
API_BASE_URL="https://api.openai.com/v1"
OPENAI_REASONING_EFFORT="medium"
SYSADMIN_ENV_SERVER_URL="ws://127.0.0.1:8000/ws"
SYSADMIN_ENV_HEALTHCHECK_URL="http://127.0.0.1:8000/health"
SYSADMIN_ENV_TASKS_URL="http://127.0.0.1:8000/tasks"
SYSADMIN_ENV_TASK_ID=""
MODEL_API_TIMEOUT_SECONDS="20"
EPISODE_TIMEOUT_SECONDS="600"
```

notes:

- `HF_TOKEN` is the first credential name the baseline checks, followed by `OPENAI_API_KEY` and `API_KEY`.
- `SYSADMIN_ENV_TASK_ID=""` means “run all tasks returned by `/tasks` in order”.
- `API_BASE_URL` may point to any openai-compatible endpoint.
- the script writes `[START]`, `[STEP]`, and `[END]` records to stdout and diagnostics to stderr.

## validation flow

there are three useful validation layers.

### 1. python tests

run the full suite:

```bash
uv run pytest -q
```

for packaging, server-contract, and scoring-focused checks, a narrower command is:

```bash
uv run pytest -q tests/test_packaginge.py tests/test_server.py tests/test_rewards.py tests/test_inferenxe.py
```

### 2. openenv manifest validation

```bash
openenv validate
```

this checks the submission structure and endpoint declarations from `openenv.yaml`.

### 3. end-to-end submission helper

the repository includes an exact pre-submission helper script:

```bash
bash scripts/validate-submission.sh https://your-space.hf.space .
```

or, from the repository root:

```bash
bash scripts/validate-submission.sh https://your-space.hf.space
```

the script performs four checks in sequence:

1. `GET <space>/health`
2. `POST <space>/reset`
3. local `docker build`
4. local `openenv validate`

use the runtime url ending in `.hf.space`, not the repository page url under `huggingface.co/spaces/...`.

## docker and deployment flow

### local docker build

```bash
docker build -t sysadmin-env .
docker run --rm -p 18000:8000 sysadmin-env
curl http://127.0.0.1:18000/health
curl http://127.0.0.1:18000/tasks
```

both `Dockerfile` and `server/Dockerfile`:

- start from `python:3.11-slim`
- install `bubblewrap`, `fuse-overlayfs`, `procps`, `iputils-ping`, `findutils`, and `curl`
- install `uv`
- copy `pyproject.toml` and `uv.lock`
- run `uv sync --locked --no-dev --no-install-project`
- copy the project files including `README.md`, root shims, `server/`, `sysadmin_env/`, and `assets/`
- run `uv sync --locked --no-dev --no-editable`
- start the environment with `uv run server --host 0.0.0.0 --port 8000`

### hugging face deployment

the repository is prepared for a hugging face docker space.

key points:

- the readme front matter declares `sdk: docker`
- `Dockerfile` is suitable for space runtime startup
- `openenv.yaml` declares `inference.py` as the benchmark entrypoint and `server.app:app` as the server entrypoint
- the root shims (`client.py`, `models.py`, `__init__.py`) and `server/Dockerfile` are present because openenv repository checks expect this structure after an `openenv init` style workflow

typical flow:

1. build and test locally
2. run `openenv validate`
3. push the repository or space update
4. wait for the hugging face space to become healthy
5. run `bash scripts/validate-submission.sh https://your-space.hf.space .`
6. run your agent against the live deployment via `inference.py`

### openenv submission commands

```bash
openenv validate
openenv push
```

this repository keeps the mirrored build assets and root shims needed for that workflow.

## mathematical summary of each task’s total raw return

ignoring catastrophic termination, the raw episode return for each task can be written as:

```text
R = H_final + K_total - 0.01 * n
```

where `n` is the number of executed steps.

for the fully solved case (`H_final = 1.0`):

| task | fully solved raw return |
| --- | --- |
| `nginx_crash` | `R = 1.0 + K_nginx - 0.01n`, where `0 <= K_nginx <= 0.21` |
| `disk_full` | `R = 1.0 + K_disk - 0.01n`, where `0 <= K_disk <= 0.22` |
| `network_broken` | `R = 1.0 + K_net - 0.01n`, where `0 <= K_net <= 0.28` |

the score reported by `inference.py` is then:

```text
score = min(max(R, 0.0), 1.0)
```

so the benchmark strongly rewards:

- solving the task at all
- gathering useful evidence without repeating it
- reaching the repair quickly
- avoiding destructive commands entirely

## limitations and portability notes

### overlay mount constraints on hugging face and other managed runtimes

managed container platforms often restrict privileged mount operations. in practice, hugging face docker spaces may not allow kernel overlay mounts, and some environments may also lack a usable `fuse-overlayfs` path.

`sysadmin_env/overlayfs.py` handles this explicitly:

1. try kernel overlayfs
2. if that fails, try `fuse-overlayfs`
3. if that also fails, use a plain directory copy fallback

the fallback is important because it preserves correctness even when the faster mount strategies are unavailable.

### what the copy fallback means

in copy mode:

- the prepared lower filesystem is copied into the merged runtime directory
- resets rebuild that merged directory by copying from the lowerdir again
- the environment remains deterministic and functional
- resets are typically slower than true overlay copy-on-write resets

this is a deliberate portability tradeoff: the benchmark prefers “runs correctly in restricted environments” over “requires privileged overlay support”.

### additional candid limitations

- the tasks are realistic but still simplified; they use stub executables rather than full linux services.
- grading is based on explicit filesystem state rather than black-box network/service behavior.
- the baseline `success` flag in `inference.py` is a client summary heuristic, not an authoritative server-side evaluation primitive.
- the environment currently models exactly three tasks; expanding benchmark breadth would require additional task modules and graders.

## practical quickstart

if you just want the shortest useful path:

```bash
uv sync --extra dev
uv run server --host 0.0.0.0 --port 8000
```

in another shell:

```bash
uv run python inference.py
```

before submission:

```bash
openenv validate
bash scripts/validate-submission.sh https://your-space.hf.space .
```

that sequence exercises the main round 1 path from local development to deployment validation.
