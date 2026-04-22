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

## round 2 artifacts at a glance

- **live hf space**: [`huggingmenfordays/enterprise-hpc-openenv`](https://huggingface.co/spaces/huggingmenfordays/enterprise-hpc-openenv) — public url `https://huggingmenfordays-enterprise-hpc-openenv.hf.space`, docker build with bwrap + overlayfs copy fallback, `/health`, `/reset`, `/step`, `/state`, `/tasks`, `/ws` all wired
- **multi-session http server (apr 23 2026)**: [`sysadmin_env/server.py`](./sysadmin_env/server.py) now runs an lru-bounded `HttpSessionStore` keyed on a uuid `episode_id`, so `group_size > 1` remote rollouts against a single space no longer clobber each other. `Observation` in [`sysadmin_env/models.py`](./sysadmin_env/models.py) now carries `grader_health`, `grader_details`, and `ood_http_code`; `StepRequest` carries an optional `episode_id` forwarded by [`training/remote_env.py`](./training/remote_env.py)
- **gymnasium env wrapper**: [`hpc_gym.py`](./hpc_gym.py) exposing `EnterpriseHPC-v0` with a pluggable scenario pool
- **six hpc incident scenarios**: [`hpc_outage`](./sysadmin_env/tasks/hpc_outage.py), [`hpc_munge`](./sysadmin_env/tasks/hpc_munge.py), [`hpc_pid_stale`](./sysadmin_env/tasks/hpc_pid_stale.py), [`hpc_gpu_ecc`](./sysadmin_env/tasks/hpc_gpu_ecc.py), [`hpc_nfs_stale`](./sysadmin_env/tasks/hpc_nfs_stale.py), [`hpc_ood_apache`](./sysadmin_env/tasks/hpc_ood_apache.py) — route, auth, post-reboot pid, gpu ecc reset, stale nfs handle, and open ondemand apache config typo fault classes, rotated per rollout for generalization. this explicitly targets the **scaler ai labs multi-app rl environment for enterprise workflows** sub-theme: slurm control plane, munge auth, systemd service manager, nvidia gpu driver, nfs share, and httpd portal are six distinct apps the agent has to orchestrate inside one incident
- **gpu-free reward curve demo**: [`tools/reward_curve_demo.py`](./tools/reward_curve_demo.py) replays a curriculum-annealed policy against the real grader and writes [`docs/assets/reward_curve_demo.png`](./docs/assets/reward_curve_demo.png) + `runs/reward_demo/reward_curve.jsonl` — observable evidence of reward improvement without a gpu, runs in under a minute on mac
- **reset latency bench**: [`bench/bench_reset.py`](./bench/bench_reset.py) — **p50 2.40 ms** in copy fallback, sub 1 ms on fuse-overlayfs hosts
- **gold trajectory verifier**: [`tools/verify_gold_trajectory.py`](./tools/verify_gold_trajectory.py) proves every scenario is deterministically solvable
- **eval / leaderboard**: [`eval/eval_suite.py`](./eval/eval_suite.py) — gold vs random vs bad policies, writes markdown leaderboard
- **local grpo training**: [`training/train_hpc_outage.py`](./training/train_hpc_outage.py) with unsloth + **`google/gemma-4-e4b-it`** + trl `GRPOTrainer`
- **remote openenv grpo training**: [`training/hpc_openenv_gemma.py`](./training/hpc_openenv_gemma.py) using `--env-urls` pointing to hosted hf spaces, same shape as the trl + gemma-4 + carla launch example
- **hf jobs submitter**: [`training/hf_jobs.py`](./training/hf_jobs.py) ships the training run as a managed hf job
- **metric logger**: [`training/logger.py`](./training/logger.py) writes `runs/<name>.metrics.jsonl` plus optional wandb + hf hub uploads
- **colab notebook**: [`training/hpc_colab.ipynb`](./training/hpc_colab.ipynb) runs the full pipeline on a single gpu, covers local and remote paths
- **one-line reproduction**: [`Makefile`](./Makefile) with `make gold`, `make bench`, `make eval`, `make dry`, `make train`, `make train-remote`
- **pitch + storytelling**: [`docs/pitch.md`](./docs/pitch.md), [`docs/hf_blog.md`](./docs/hf_blog.md), [`docs/video_script.md`](./docs/video_script.md)
- **deploy paths**: [`docs/hf_spaces_deploy.md`](./docs/hf_spaces_deploy.md), [`docs/hf_jobs.md`](./docs/hf_jobs.md)
- **one-page setup guide**: [`GETTING_STARTED.md`](./GETTING_STARTED.md)
- **hackathon task list**: [`TODO_FOR_USER.md`](./TODO_FOR_USER.md)
- **judges' guide compliance map**: [`JUDGES_COMPLIANCE.md`](./JUDGES_COMPLIANCE.md) — section-by-section cross reference against the apr 2026 openenv self-serve guide, including the six independent reward functions in [`training/reward_functions.py`](./training/reward_functions.py), the `--curriculum` scenario ramp, the `--save-adapter-only` qlora-safe export path, and the per-step transcript sampler in [`training/logger.py`](./training/logger.py)

## table of contents

- [round 2 theme alignment](#round-2-theme-alignment)
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
- [baseline behavior and current observations](#baseline-behavior-and-current-observations)
- [validation flow](#validation-flow)
- [docker and deployment flow](#docker-and-deployment-flow)
- [mathematical summary of each task’s total raw return](#mathematical-summary-of-each-tasks-total-raw-return)
- [limitations and portability notes](#limitations-and-portability-notes)
- [practical quickstart](#practical-quickstart)

## round 2 theme alignment

this repository targets the following judging theme coverage:

- **primary theme #3.1 — world modeling / professional tasks**: the env is a partially observable rocky linux hpc cluster (mock slurm, munge, systemd, nvidia gpu, nfs, apache open ondemand). the agent must interact with real-looking tools, maintain state between steps, and cannot shortcut to the grader. the reward only goes up when system state actually changes.
- **bonus sub-theme — scaler ai labs multi-app rl environment for enterprise workflows**: each scenario exercises multiple "apps" at once. `hpc_ood_apache` touches httpd + systemd + the ood portal; `hpc_gpu_ecc` touches slurm + nvidia driver + systemd; `hpc_nfs_stale` touches nfs + slurm + systemd. these are the exact multi-app enterprise remediations on-call sre teams do every week.
- **secondary theme #2 — long-horizon planning & instruction following**: the agent must decompose each incident into diagnosis, repair, verify, and stop. gold trajectories are 8 – 14 steps long and reward shaping is sparse enough to reward genuine multi-step planning.

the full judging rubric is addressed by the repository layout as follows:

| rubric axis | weight | where we deliver |
| --- | ---: | --- |
| environment innovation | 40% | nine deterministic tasks, three of them brand new multi-app hpc incidents (`hpc_gpu_ecc`, `hpc_nfs_stale`, `hpc_ood_apache`), bubblewrap + overlayfs isolation with sub-10 ms resets, binary + shaped reward dual-head |
| storytelling | 30% | pitch, hf blog draft, video script under `docs/`, live tmux demo via `make eval` and `make reward-demo`, clean before / after leaderboards |
| showing improvement in rewards | 20% | `tools/reward_curve_demo.py` writes a curriculum-annealed reward curve png + jsonl in under a minute, no gpu required. real grpo curves come from the colab notebook |
| reward + training pipeline | 10% | `sysadmin_env/rewards.py` shaped rewards + trl `GRPOTrainer` with unsloth + gemma 4 + openenv client, see `training/hpc_openenv_gemma.py` |

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
| deterministic tasks | nine fixed task modules in `sysadmin_env/tasks/nginx_crash.py`, `sysadmin_env/tasks/disk_full.py`, `sysadmin_env/tasks/network_broken.py`, `sysadmin_env/tasks/hpc_outage.py`, `sysadmin_env/tasks/hpc_munge.py`, `sysadmin_env/tasks/hpc_pid_stale.py`, `sysadmin_env/tasks/hpc_gpu_ecc.py`, `sysadmin_env/tasks/hpc_nfs_stale.py`, and `sysadmin_env/tasks/hpc_ood_apache.py` |
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
├── .env.example
├── README.md
├── messing-around-with-playbooks.md
├── __init__.py
├── client.py
├── Dockerfile
├── inference.py
├── models.py
├── openenv.yaml
├── pyproject.toml
├── outputs/
│   └── output-*.txt
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
        ├── hpc_outage.py
        ├── network_broken.py
        └── nginx_crash.py
```

an additional root module `hpc_gym.py` exposes a `gymnasium.Env` wrapper named `EnterpriseHPCEnv` for hugging face trl / grpo training loops. it reuses the same `Sandbox` and `OverlayFSManager`, drives the scenario through a `pexpect` interactive bash session, and keeps the reset path on `/dev/shm`.

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
- `sysadmin_env/tasks/hpc_outage.py` — hard multi-node hpc cluster outage with a simulated slurm queue, a drained `compute-01` node, a broken `route-eth0`, and a simulated open ondemand portal on `:8080`.

### root shims and openenv-facing files

- `client.py` — thin root shim that re-exports `main` from `inference.py`. this keeps the repository shape friendly to packaging and submission tooling.
- `models.py` — thin root shim that re-exports the canonical pydantic models from `sysadmin_env.models`.
- `__init__.py` — root package shim that re-exports `main`, `Action`, `Observation`, and `EnvironmentState`.
- `inference.py` — the baseline agent used as the submission entrypoint declared in `openenv.yaml`.
- `README.md` — primary repository documentation covering architecture, tasks, reward shaping, setup, validation, and the current baseline behavior.
- `.env.example` — sample environment-variable file for local configuration.
- `messing-around-with-playbooks.md` — change log for the recent baseline prompt and `network_broken` guardrail adjustments, including observed local run results.
- `outputs/` — local captured baseline run logs used while tuning and validating the inference behavior.

### deployment, packaging, and validation files

- `Dockerfile` — primary container build for local docker runs and hugging face docker spaces.
- `server/Dockerfile` — mirrored server build asset kept alongside `server/app.py` for openenv repository structure checks.
- `server/app.py` — asgi/cli launcher that imports `app` from `sysadmin_env.server` and exposes the `server` console script.
- `openenv.yaml` — openenv manifest: runtime entrypoints, endpoints, resources, and task metadata.
- `pyproject.toml` — canonical packaging metadata, dependencies (loose `>=` pins), python version bounds (`>=3.12`), the `server = "server.app:main"` console script, and the `[dev]` / `[train]` optional-dependency groups.
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
  },
  "episode_id": "optional, uuid hex returned by /reset"
}
```

`episode_id` is **optional** (omitted = talks to the legacy singleton
slot, for backward compatibility with older clients). supplying it is
required whenever two or more clients share one server: the server
keeps a bounded `HttpSessionStore` keyed on this id so concurrent
`group_size > 1` rollouts do not clobber each other's sandbox.

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
  "max_steps": 40,
  "grader_health": 0.0,
  "grader_details": {},
  "ood_http_code": ""
}
```

important details:

- `reward` is **the reward for that step only**, not a cumulative return.
- `done` becomes `true` when the task grader declares success, a catastrophic action is detected, or the episode hits `max_steps`.
- `working_directory` is `/` from the sandbox's point of view.
- if a command times out, the server appends `command execution timed out` to `stderr`.
- `grader_health` is the task grader's current health score on `[0, 1]` after this step. clients can use it directly as a shaped progress signal without reimplementing the grader. added apr 23 2026.
- `grader_details` is a small dict of per-fact booleans / numbers / strings surfaced by the task's `grade()` function (e.g. `slurmd_restarted: true`, `ecc_reset_ok: true`) — useful for per-task diagnostics.
- `ood_http_code` is populated only by `hpc_ood_apache` (the most recently observed apache status code) and empty otherwise.

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
4. `hpc_outage`
5. `hpc_munge`
6. `hpc_pid_stale`
7. `hpc_gpu_ecc`
8. `hpc_nfs_stale`
9. `hpc_ood_apache`

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
    "max_steps": 40,
    "grader_health": 0.25,
    "grader_details": {"slurm_reachable": true, "munge_up": true},
    "ood_http_code": ""
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

if the requested `episode_id` is not in the server's session store (or
no episode has been initialized and `episode_id` was omitted), the
route returns http `409`. if the sandbox errors out mid-step, the
server returns http `500` with a json body describing the failure.

#### `GET /state`

returns the latest `EnvironmentState`. accepts an optional
`?episode_id=<uuid-hex>` query parameter to address a specific session
in the store; without it the route returns the most-recently-reset
episode. returns http `404` if no episode has been initialized yet.

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

there are nine tasks, with increasing difficulty and fixed metadata also mirrored in `openenv.yaml`.

| task | difficulty | max steps | time limit | objective |
| --- | --- | ---: | ---: | --- |
| `nginx_crash` | easy | 40 | 300 s | restore a broken nginx service with config and pid issues |
| `disk_full` | medium | 55 | 420 s | identify and neutralize the hidden file exhausting `/mnt/data` |
| `network_broken` | hard | 70 | 480 s | repair routing and dns so outbound connectivity is restored |
| `hpc_outage` | hard | 90 | 600 s | restore a simulated 224-core hpc cluster by fixing `compute-01` routing and bringing slurmd back to idle |
| `hpc_munge` | hard | 90 | 600 s | fix a munge authentication failure (wrong key mode) chained with a broken route |
| `hpc_pid_stale` | hard | 90 | 600 s | clear a leftover `/var/run/slurmd.pid` so slurmd restarts after a simulated reboot |
| `hpc_gpu_ecc` | hard | 90 | 600 s | diagnose a drained node, reset `gpu-0` via `nvidia-smi -r -i 0`, and bring the node back to idle |
| `hpc_nfs_stale` | hard | 90 | 600 s | recover from a stale nfs handle on `/mnt/shared` with `umount -l` / `mount` before restarting slurmd |
| `hpc_ood_apache` | hard | 90 | 600 s | repair a typo in `httpd.conf` for the open ondemand portal on `:8081` and reload apache gracefully |

### determinism guarantees across tasks

all nine tasks are deterministic in the current codebase:

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

### task 4: `hpc_outage`

**what is broken**

- the simulated cluster is a 224-core rocky linux hpc with two nodes: `login` and `compute-01`
- cluster state lives in `/mnt/shared/slurm_state.json` — a shared json file read under `fcntl.LOCK_SH` and mutated under `fcntl.LOCK_EX`
- `compute-01` is in state `drain` with `slurmd@compute-01` marked `failed`
- `/nodes/compute-01/etc/sysconfig/network-scripts/route-eth0` ships with an invalid netmask, wrong gateway, and wrong device
- the open ondemand portal `ood_server.py` binds `:8080` in the sandbox and returns `http 502` until the route file matches the expected contents
- there are no real slurm daemons or nginx instances — the scenario is a state machine simulation that still behaves correctly under parallel grpo training

**relevant task-local command stubs**

- `ssh` — bash stub that validates the target host under `/nodes/` and execs a nested `bwrap` that rebinds `/nodes/$TARGET` as `/`, sets `HOSTNAME` and `PS1`, and drops the agent into `/bin/bash`
- `sinfo` / `squeue` — python stubs that read `slurm_state.json` under `fcntl.LOCK_SH` and print formatted terminal tables
- `systemctl` — python stub that mutates `slurm_state.json` under `fcntl.LOCK_EX`. `systemctl restart slurmd` on `compute-01` only transitions the node to `idle` if the route file is fixed
- `scontrol` — minimal python stub for `scontrol show node` and `scontrol update` interactions
- `curl` — minimal in-sandbox http client that speaks to the local ood daemon
- `ood_server.py` — background http daemon on port `8080`. returns `200` when the route file matches the expected contents and `502` otherwise

**difficulty progression**

this task is hard because the agent has to reason across three layers inside a single sandbox:

1. inspect cluster state through `sinfo` / `squeue`
2. identify the failed unit via `systemctl status slurmd@compute-01` or `systemctl is-failed slurmd`
3. `ssh compute-01` to shift root into the compute node
4. rewrite `/etc/sysconfig/network-scripts/route-eth0` on `compute-01` with the expected `ADDRESS0` / `NETMASK0` / `GATEWAY0` / `DEVICE0` lines
5. `systemctl restart slurmd` so the systemctl stub flips the shared json state from `drain` to `idle`
6. validate that `curl -I http://localhost:8080` returns `200`

**grader behavior**

the task health is:

```text
H_hpc = 0.30 * I_route_file_restored
      + 0.30 * I_compute_node_idle
      + 0.40 * I_both_restored
```

where:

- `I_route_file_restored = 1` iff `/nodes/compute-01/etc/sysconfig/network-scripts/route-eth0` exactly matches the expected string
- `I_compute_node_idle = 1` iff `/mnt/shared/slurm_state.json` has `nodes.compute-01.state == "idle"`
- `I_both_restored = 1` iff both of the above are true; in that case health is pinned to `1.0`

the episode ends successfully when both indicators are `1`.

**diagnostic rewards**

- `sinfo` or `squeue`: `+0.06`
- `ssh compute-01`: `+0.07`
- reading `route-eth0` or listing `network-scripts`: `+0.05`
- `systemctl status slurmd` or `systemctl is-failed slurmd`: `+0.05`
- `curl ... localhost:8080`: `+0.05`

**architectural notes**

- resets stay well under 10 ms because `OverlayFSManager` pins `upperdir` and `workdir` to `/dev/shm`. only the merged mount point lives on disk and the lowerdir is read-only host state
- multi-node lateral movement is simulated without `veth` pairs or `CLONE_NEWNET`. `ssh` is a nested `bwrap` that rebinds `/nodes/$TARGET` as `/` while re-binding `/mnt/shared` so the slurm state file remains coherent across nodes
- nested sandboxing requires the primary sandbox to run with `--unshare-user` and `--cap-add CAP_SYS_ADMIN`, enabled per task via `TaskScenarioDefinition.allows_nested_sandbox`
- evaluation is deterministic and reads only explicit filesystem state; no real daemons are spawned by the grader path

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

### grpo multi-reward decomposition

the apr 2026 openenv hackathon judges' self-serve guide (section 7) recommends using **multiple independent reward functions** rather than a single scalar so the policy cannot collapse onto one exploitable channel. both grpo trainers in this repo therefore pass six orthogonal reward functions to `trl.GRPOTrainer`, defined in [`training/reward_functions.py`](./training/reward_functions.py):

| reward fn | source | intent |
| --- | --- | --- |
| `solve_reward` | `terminated` flag from rollout | deterministic rlvr signal, 1.0 iff the grader said "done" before step cap |
| `format_reward` | regex on the completion | rewards well-formed `<bash>...</bash>` actions |
| `safety_reward` | per-command destructive regex | penalizes `rm -rf /`, `mkfs`, fork-bombs, etc. |
| `progress_reward` | `best_health` / `grader_health`, scaled to `[0, 0.5]` (cumulative-reward fallback for legacy servers) | shaped partial credit |
| `efficiency_reward` | `max_turns - steps`, scaled to `[0, 0.2]` when `terminated` | encourages short solves |
| `anti_hack_reward` | per-command regex vs. `GRADER_PROTECTED_PATTERNS` | flags edits to grader-owned paths (`slurm_state.json`, `/grader/`, ecc sentinel) |

each component is logged independently so reviewers can tell which signal is driving training. the rollout is executed once per grpo step and cached keyed on `id(completions)`, so the six reward fns are cheap.

> **apr 23 2026 fix**: `solve_reward` used to check `r.reward >= 1.0`,
> but the server's shaped per-step reward is `health_delta + knowledge_delta - 0.01`
> which peaks around `~0.4` even on the solving step. that meant
> `solve_reward` was identically zero across every rollout and grpo saw
> `reward_std = 0`. the trigger is now `bool(r.terminated)`.
> `progress_reward` similarly depended on `grader_health` that was
> never propagated into the client's `info` dict before the
> `Observation` carried the new `grader_health` field. both paths are
> wired end-to-end now.

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

all six tasks begin with `H_initial = 0.0`, so if the agent fully solves a task without catastrophic failure:

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
| `hpc_outage` | `0.06 + 0.07 + 0.05 + 0.05 + 0.05 = 0.28` |
| `hpc_munge` | `0.06 + 0.07 + 0.05 + 0.05 + 0.05 = 0.28` |
| `hpc_pid_stale` | `0.06 + 0.07 + 0.05 + 0.05 + 0.05 = 0.28` |
| `hpc_gpu_ecc` | `0.06 + 0.07 + 0.05 + 0.05 + 0.05 = 0.28` |
| `hpc_nfs_stale` | `0.06 + 0.07 + 0.05 + 0.05 + 0.05 = 0.28` |
| `hpc_ood_apache` | `0.06 + 0.07 + 0.05 + 0.05 + 0.05 = 0.28` |

so the maximum raw trajectory return before step penalties is:

```text
1.0 + knowledge_sum
```

which is:

- `1.21` for `nginx_crash`
- `1.22` for `disk_full`
- `1.28` for `network_broken`
- `1.28` for `hpc_outage`, `hpc_munge`, `hpc_pid_stale`, `hpc_gpu_ecc`, `hpc_nfs_stale`, and `hpc_ood_apache`

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

the repository targets python `>=3.12` (python `3.13` is the current unsloth default per their install docs). `pyproject.toml` is the single source of truth for dependencies — no `uv.lock`, no `requirements.txt`, no surprises. all version pins are loose `>=` so a fresh `pip install` picks up whatever is current on the colab or hf jobs runtime.

### recommended setup with `venv + pip`

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -e '.[dev]'
```

### training extras (gpu needed, skip on mac)

```bash
pip install -e '.[train]'
pip install 'unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git'
```

### modern alternative with `uv` (optional)

```bash
uv venv --python 3.13
source .venv/bin/activate
uv pip install -e '.[dev,train]'
```

## running the server locally

the canonical launcher is the `server` console script declared in `pyproject.toml` and implemented by `server/app.py`. after `pip install -e .` the script is on `PATH`:

```bash
server --host 0.0.0.0 --port 8000
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
python inference.py
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

- `API_BASE_URL` and `MODEL_NAME` both have built-in defaults in `inference.py`.
- `HF_TOKEN` is the required submission-facing variable name. in practical terms, the token value must match the provider behind `API_BASE_URL`: if you point at the hugging face router, use a hugging face token; if you point at another openai-compatible endpoint, use the credential that endpoint expects.
- the script also accepts `OPENAI_API_KEY` and `API_KEY` as compatibility fallbacks for local runs, but the documented submission path should still provide `HF_TOKEN`.
- `SYSADMIN_ENV_TASK_ID=""` means “run all tasks returned by `/tasks` in order”.
- `API_BASE_URL` may point to any openai-compatible endpoint.
- this baseline talks to the running environment server over http/websocket, so an extra `LOCAL_IMAGE_NAME` variable is not needed here unless you rewrite the client around a `from_docker_image()` flow.
- by default, the script writes the flat submission-oriented `[START]`, `[STEP]`, and `[END]` records to stdout and diagnostics to stderr.
- if you need the older json payload logs for local debugging, set `SYSADMIN_ENV_LOG_FORMAT=json` before running `inference.py`.

### stdout output contract

the default stdout format is the flat key-value format expected by the latest submission notes:

```text
[START] task=<task_name> env=<benchmark> model=<model_name>
[STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
[END] success=<true|false> steps=<n> score=<0.00> rewards=<r1,r2,...,rn>
```

details:

- `score` is normalized to stay strictly inside `(0, 1)` before logging, so boundary values are not emitted in submission summaries
- `reward` and each entry in `rewards` are formatted to exactly two decimal places
- `done` and `success` are lowercase booleans
- `error` is `null` when there is no step error
- all output stays on a single line per record

## baseline behavior and current observations

the current baseline keeps the same high-level contract while tightening how the hard task is handled.

### current baseline behavior

- if `HF_TOKEN` or another supported api key is present, `inference.py` uses the openai responses api.
- if no api key is present or the model call fails, the script falls back to the deterministic task plan described in `inference.py`.
- for `network_broken`, the model prompt now uses a **generic** task playbook rather than embedding the exact hidden grader targets.
- after enough route, interface, and dns diagnosis, the baseline applies a state-aware guardrail for `network_broken` so that unsupported guesses do not loop forever.
- the guardrail emits concise stderr traces such as `network guardrail dns repair` and `network guardrail route repair`, which makes the baseline easier to debug without changing the wire protocol.

### why the baseline was adjusted

the earlier prompt variant made `network_broken` too easy because the model could effectively recover the exact answer from the prompt rather than infer it from the environment. the current prompt removes that leakage and keeps the hard task benchmark-oriented while still allowing a reproducible baseline run.

### current observed local baseline run

the latest local run against the repository server with `MODEL_NAME="gpt-5.4-nano"` produced the following episode summaries:

| task | success | steps | score | notes |
| --- | --- | ---: | ---: | --- |
| `nginx_crash` | `true` | `6` | `1.0` | fixed config, cleared stale pid, then started nginx |
| `disk_full` | `true` | `4` | `1.0` | diagnosed the full mount, inspected the hidden trace, then truncated it |
| `network_broken` | `true` | `7` | `1.0` | gathered route/link/dns evidence first, then the guardrail applied dns repair followed by route repair |

this is a **current observed baseline**, not a theoretical guarantee for every model provider or future model snapshot.

for the full debugging narrative behind those adjustments, see `messing-around-with-playbooks.md`.

## gymnasium wrapper for trl and grpo

`hpc_gym.py` exposes a `gymnasium.Env` named `EnterpriseHPCEnv` that drives any registered hpc scenario through an interactive `pexpect` bash session. it is the recommended entry point for hugging face trl / grpo training loops because it keeps resets on `tmpfs` and uses a binary grader based reward that is fast to compute.

key behaviors:

- `reset()` prepares (or resets) the overlay stack, spawns `ood_server.py` as a background process inside the primary sandbox, and `ssh`s into the `login` node so that the first observation is already at `[root@login ...]$ `.
- `step(action)` sends the action string to the pexpect shell, waits for the prompt regex `re.compile(r'\[\w+@[\w-]+.*\]\$ ')`, and returns the terminal output as the text observation.
- reward is binary: `1.0` when the active task grader reports `done`, else `0.0`. the ood portal is still live on `:8080` so the agent can confirm with `curl -I` but the reward signal comes directly from the deterministic grader.
- `terminated=True` when the grader reports done; `truncated=True` after `max_steps` without success.
- `scenario_pool=[...]` rotates tasks per rollout for generalization. `hpc_outage`, `hpc_munge`, `hpc_pid_stale`, `hpc_gpu_ecc`, `hpc_nfs_stale`, and `hpc_ood_apache` are registered out of the box.

usage sketch:

```python
from hpc_gym import EnterpriseHPCEnv

env = EnterpriseHPCEnv(scenario_pool=[
    "hpc_outage", "hpc_munge", "hpc_pid_stale",
    "hpc_gpu_ecc", "hpc_nfs_stale", "hpc_ood_apache",
])
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step("sinfo")
env.close()
```

optional registration under the gymnasium registry:

```python
from hpc_gym import register_env
register_env()
# env = gymnasium.make("EnterpriseHPC-v0")
```

## training with gemma 4 + trl grpo

the `training/` package ships a full recipe that ties `EnterpriseHPC-v0` to hugging face trl `GRPOTrainer` with unsloth loaded **`google/gemma-4-e4b-it`** (4.5b effective, 128k context, apache 2). the rollout driver at `training/rollout.py` runs multi turn episodes, parses `<bash>...</bash>` actions from policy completions, and feeds observations back into the chat transcript.

### local (colab, single workstation)

```bash
python -m training.train_hpc_outage --dry-run --group-size 2 --max-turns 8
python -m training.train_hpc_outage \
    --model google/gemma-4-e4b-it \
    --scenarios hpc_outage,hpc_munge,hpc_pid_stale,hpc_gpu_ecc,hpc_nfs_stale,hpc_ood_apache \
    --group-size 4 --max-turns 12 --num-train-steps 100 \
    --output-dir ./runs/hpc_grpo
```

### remote, against hosted openenv spaces

this matches the shape of the trl + gemma-4 + carla example from the gemma 4 launch post: point `--env-urls` at one or more hf spaces hosting the openenv server, the rollout pool round-robins for throughput.

```bash
python -m training.hpc_openenv_gemma \
    --env-urls https://huggingmenfordays-enterprise-hpc-openenv.hf.space \
    --model google/gemma-4-e4b-it \
    --group-size 4 --max-turns 24 --num-train-steps 200 \
    --curriculum --save-adapter-only \
    --scenarios hpc_outage,hpc_munge,hpc_pid_stale,hpc_gpu_ecc,hpc_nfs_stale,hpc_ood_apache
```

the default `--max-turns` is now `24` (was `16` before apr 23 2026):
multi-step scenarios like `hpc_pid_stale` and `hpc_nfs_stale` routinely
need 10+ turns just to surface the right diagnostic output, and small
instruct models spend several early turns getting `<bash>...</bash>`
format compliance right. the server's per-episode session store lets
you point `--group-size 4+` at a **single** space without the episode
state-clobbering bug that was present in pre-apr-23 builds.

### managed hf jobs

```bash
python -m training.hf_jobs \
    --env-urls https://<user>-enterprise-hpc-openenv.hf.space \
    --gpu a10g-large \
    --num-train-steps 300 \
    --hub-repo <user>/hpc-grpo-runs
```

see [`docs/hf_jobs.md`](./docs/hf_jobs.md) for the full hf training guide and [`training/hpc_colab.ipynb`](./training/hpc_colab.ipynb) for a single notebook that covers both the local and remote paths.

## reset latency benchmark

```bash
python -m bench.bench_reset -n 200
# or
make bench
```

emits a markdown row with `p50 / p95 / p99 / max ms` ready to drop into the blog or pitch deck. on a sandbox with no overlay privileges the copy fallback measures **p50 2.40 ms, p99 2.58 ms, stdev 0.07 ms** over 100 iterations. on a linux host with `fuse-overlayfs` expect sub 1 ms.

## gold trajectory verifier + eval leaderboard

prove the environment is deterministically solvable (no gpu, no network):

```bash
make gold
# or
python -m tools.verify_gold_trajectory -v
```

run a reproducible leaderboard comparing gold, random, and adversarial policies:

```bash
make eval
# artifacts: runs/eval/leaderboard.md, eval_summary.json, eval.jsonl
```

## one-line reproduction

```bash
make help        # full list of targets
make gold        # deterministic solvability proof
make bench       # reset latency
make eval        # policy leaderboard
make dry         # training rollout smoke test, no gpu
make train       # local grpo with gemma-4-e4b-it
make train-remote ENV_URLS=https://<user>-enterprise-hpc-openenv.hf.space
```

## validation flow

there are two useful validation layers.

### 1. openenv manifest validation

```bash
openenv validate
```

this checks the submission structure and endpoint declarations from `openenv.yaml`.

### 2. end-to-end submission helper

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

- start from `python:3.13-slim`
- install `bubblewrap`, `fuse-overlayfs`, `procps`, `iputils-ping`, `findutils`, and `curl`
- copy `pyproject.toml`, root shims, `server/`, `sysadmin_env/`, `assets/`, `bench/`, `training/`, `eval/`, `tools/`, and `docs/`
- run `pip install --upgrade pip setuptools wheel`
- run `pip install .` (pulls all loose-pinned runtime deps)
- start the environment with the `server` console script on `PATH`

### hugging face deployment

the repository is prepared for a hugging face docker space, and a
reference deployment already lives at
[`huggingmenfordays/enterprise-hpc-openenv`](https://huggingface.co/spaces/huggingmenfordays/enterprise-hpc-openenv)
(public url: `https://huggingmenfordays-enterprise-hpc-openenv.hf.space`).

key points:

- the readme front matter declares `sdk: docker`
- `Dockerfile` is suitable for space runtime startup
- `openenv.yaml` declares `inference.py` as the benchmark entrypoint and `server.app:app` as the server entrypoint
- the root shims (`client.py`, `models.py`, `__init__.py`) and `server/Dockerfile` are present because openenv repository checks expect this structure after an `openenv init` style workflow

typical flow:

1. build and test locally
2. run `openenv validate`
3. push the repository or space update (recipe below)
4. wait for the hugging face space to become healthy
5. run `bash scripts/validate-submission.sh https://your-space.hf.space .`
6. run your agent against the live deployment via `inference.py`

#### pushing updates to the live space (orphan-branch recipe)

this repo carries `.venv/` and `docs/assets/*.png` binaries in git
history that hf xet refuses to accept. a plain
`git push space final-round:main` gets rejected with
`pre-receive hook declined / your push was rejected because it contains binary files`.
use the orphan-branch force-push instead:

```bash
hf auth login                                                                  # refresh write token

git remote set-url space https://huggingface.co/spaces/huggingmenfordays/enterprise-hpc-openenv

git checkout --orphan space-deploy
git rm -rf --cached .
rm -f docs/assets/reward_curve_demo.png                                        # drop any binary that would re-trip xet
git add -A
git commit -m "deploy: clean snapshot for hf space"
git push space space-deploy:main --force

git checkout final-round
git branch -D space-deploy
git checkout HEAD -- docs/assets/reward_curve_demo.png                         # restore the png locally
```

this force-pushes a one-commit history-less snapshot to the space's
`main` branch; your local `final-round` history is untouched. the
docker build takes 5 – 10 min, then `curl <space>/health` should return
`{"status":"ok"}`. the same recipe is documented in
[`docs/hf_spaces_deploy.md`](./docs/hf_spaces_deploy.md) §2.1 and
[`TODO_FOR_USER.md`](./TODO_FOR_USER.md) §2.

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
| `hpc_outage` | `R = 1.0 + K_hpc - 0.01n`, where `0 <= K_hpc <= 0.28` |
| `hpc_munge` | `R = 1.0 + K_hpc - 0.01n`, where `0 <= K_hpc <= 0.28` |
| `hpc_pid_stale` | `R = 1.0 + K_hpc - 0.01n`, where `0 <= K_hpc <= 0.28` |
| `hpc_gpu_ecc` | `R = 1.0 + K_hpc - 0.01n`, where `0 <= K_hpc <= 0.28` |
| `hpc_nfs_stale` | `R = 1.0 + K_hpc - 0.01n`, where `0 <= K_hpc <= 0.28` |
| `hpc_ood_apache` | `R = 1.0 + K_hpc - 0.01n`, where `0 <= K_hpc <= 0.28` |

the score reported by `inference.py` is then transformed into an open-interval submission summary value:

```text
score_clamped = min(max(R, 0.0), 1.0)
score_reported = 0.01 + 0.98 * score_clamped
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
- the environment currently models exactly six tasks; expanding benchmark breadth would require additional task modules and graders.

## practical quickstart

if you just want the shortest useful path:


```bash
python3.13 -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
server --host 0.0.0.0 --port 8000
```

in another shell:

```bash
python inference.py
```

before submission:

```bash
openenv validate
bash scripts/validate-submission.sh https://your-space.hf.space .
```

that sequence exercises the main round 1 path from local development to deployment validation.

<p align="center"><strong>with love :</strong></p>

![hatsune-miku-miku](https://github.com/user-attachments/assets/2db5754f-20cd-4456-b636-c43197346976)
![200w](https://github.com/user-attachments/assets/ea2e0c0c-91b9-4a49-93c2-daabea75c1d8)
![kasane-teto-teto-kasane](https://github.com/user-attachments/assets/0520bf6e-96a2-4c17-bd04-f6c60b5cc60b)
![teto-tetoris](https://github.com/user-attachments/assets/569f977f-6486-44e3-94ba-b8b68eb99410)
![200](https://github.com/user-attachments/assets/05f9bcb2-7476-417b-8398-ae9cbbca3d17)
