# judges' self-serve guide compliance map

this document cross-references the apr 2026 openenv hackathon self-serve guide
(22 sections + 58 faq entries + 59 unsloth recipe pointers) to concrete
artifacts in this repo. every section of the guide is covered here, with the
file paths, commands, and rationale a judge can follow in under five minutes.

> **tl;dr** every explicit "must do" from the guide is implemented. the only
> items the repo cannot self-complete are the two blockers tracked in
> [`TODO_FOR_USER.md`](./TODO_FOR_USER.md): a real gpu grpo training curve
> and the 90-second demo video. the live hugging face space
> (`huggingmenfordays/enterprise-hpc-openenv`) is deployed. gpu-free evidence of
> reward improvement already lives in [`docs/assets/reward_curve_demo.png`](./docs/assets/reward_curve_demo.png).

> **apr 23 2026 update**: the remote rollout pipeline was rewritten so
> `group_size > 1` against a single hf space no longer clobbers
> episode state. the server ([`sysadmin_env/server.py`](./sysadmin_env/server.py))
> now runs an lru-bounded `HttpSessionStore` keyed on a uuid
> `episode_id`; `Observation` carries `grader_health`,
> `grader_details`, and `ood_http_code`; and
> [`training/reward_functions.py`](./training/reward_functions.py) now
> triggers `solve_reward` on `terminated` (not a reward threshold) and
> consumes the propagated `grader_health` for `progress_reward`. this
> fixed a `frac_reward_zero_std = 1` stall observed on the first full
> kaggle probe run.

## 0. what you are building → environment + verifier + trainer + deployment

| layer | repo artifact |
| --- | --- |
| environment | [`sysadmin_env/`](./sysadmin_env/) fastapi server, [`hpc_gym.py`](./hpc_gym.py) gymnasium wrapper, nine scenarios in [`sysadmin_env/tasks/`](./sysadmin_env/tasks/) |
| verifier / reward | [`sysadmin_env/rewards.py`](./sysadmin_env/rewards.py), [`tools/verify_gold_trajectory.py`](./tools/verify_gold_trajectory.py), [`training/reward_functions.py`](./training/reward_functions.py) |
| trl trainer | [`training/train_hpc_outage.py`](./training/train_hpc_outage.py) local, [`training/hpc_openenv_gemma.py`](./training/hpc_openenv_gemma.py) remote via `--env-urls` |
| unsloth efficiency | `FastLanguageModel` + 4-bit qlora in both training scripts |
| openenv deploy | [`Dockerfile`](./Dockerfile), [`server/Dockerfile`](./server/Dockerfile), [`docs/hf_spaces_deploy.md`](./docs/hf_spaces_deploy.md), [`openenv.yaml`](./openenv.yaml) |

## 1. pick the right project idea (verifiable, step-by-step, hard-but-solvable)

the task is **linux hpc incident response**. the agent acts one shell command
at a time, every scenario ships with a deterministic grader, and every
scenario has a sub-14-step gold trajectory proven by
`python -m tools.verify_gold_trajectory` (`make gold`).

## 2. minimum rl loop

the loop is wired end-to-end in [`training/rollout.py`](./training/rollout.py):

1. prompt → [`training/agent_prompt.py`](./training/agent_prompt.py)
2. model generates `<bash>...</bash>`
3. action executed in `Sandbox` via bwrap + overlayfs
4. reward computed by `RewardEngine` and the six `reward_funcs`
5. grpo update in `trl.GRPOTrainer` with `num_generations=group_size`

## 3. sft vs rl

we train from `Qwen/Qwen2.5-Coder-7B-Instruct`, a code-tuned
instruction-tuned warm start, then run grpo on top. this matches the
guide's "add light formatting or task scaffolding if needed. use rl for
improvement, not as magic from scratch". the policy already emits
well-formed shell commands so grpo does not burn samples on format
discovery. any other text instruct model can be dropped in via
`--model`.

## 4 & 5. design & build the environment first

- action / observation / state types: [`sysadmin_env/models.py`](./sysadmin_env/models.py)
- `reset`, `step`, `state`, `tasks`, `health`, `ws`: [`sysadmin_env/server.py`](./sysadmin_env/server.py)
- openenv scaffold: [`openenv.yaml`](./openenv.yaml) + docker entrypoints

## 6. start simple (curriculum)

`training/train_hpc_outage.py --curriculum` and
`training/hpc_openenv_gemma.py --curriculum` unlock scenarios in three
difficulty buckets:

1. `hpc_pid_stale`, `hpc_gpu_ecc`, `hpc_ood_apache` (short, single-fix)
2. `hpc_nfs_stale` (two-step mount fix)
3. `hpc_outage`, `hpc_munge` (multi-app, branching)

this prevents the zero-reward stall the guide warns about in sections 6 and
14.

## 7. design rewards carefully (multiple independent components)

> "use multiple independent reward functions, not just one" — section 7.

the grpo trainers in this repo pass six independent reward functions to
`trl.GRPOTrainer`, all defined in [`training/reward_functions.py`](./training/reward_functions.py):

| reward fn | purpose | guide tie-in |
| --- | --- | --- |
| `solve_reward` | binary rlvr signal from grader | §7 correctness / §4 env-based reward |
| `format_reward` | rewards well-formed `<bash>` action | §7 format compliance |
| `safety_reward` | penalizes destructive shell commands | §8 reward hacking / §7 safety |
| `progress_reward` | terminal grader health, capped at 0.5 | §7 partial progress |
| `efficiency_reward` | bounded bonus for short solves | §7 timeouts / resource usage |
| `anti_hack_reward` | penalizes edits to grader-owned paths | §8 anti-cheating |

`trl` sums them into the advantage, but each column is still logged
independently so reviewers can see which signal is driving updates.

## 8. reward hacking protection

- **multiple independent signals**: see §7 above
- **locked-down execution**: [`sysadmin_env/sandbox.py`](./sysadmin_env/sandbox.py) uses bubblewrap with unshared namespaces, read-only binds, and optional `--unshare-net`
- **per-episode session isolation**: the server's `HttpSessionStore`
  keyed on uuid `episode_id` means one rollout cannot observe or
  corrupt another rollout's sandbox even when many clients share the
  same space — no cross-episode information leak
- **time limits**: `DEFAULT_STEP_TIMEOUT = 60s`, `DEFAULT_SHELL_TIMEOUT = 30s`, `max_runtime_minutes: 20` in `openenv.yaml`
- **avoid unrestricted globals**: slurm state is a json file guarded with `fcntl` locks, not a python global
- **sample + inspect**: `RewardLogger` now writes `runs/<run>/transcripts/step_NNNN.jsonl` every `transcript_sample_every` steps (default 5). see [`training/logger.py`](./training/logger.py)
- **rollback on drift**: catastrophic commands end the episode immediately with `catastrophic_penalty = -1.0` in `RewardEngine`
- **forbidden globals / protected paths**: `anti_hack_reward` checks every `<bash>` command against `GRADER_PROTECTED_PATTERNS` (includes `slurm_state.json`, `/grader/`, `ECC_RESET_SENTINEL`)

## 9. process-aware feedback

the per-step `RewardEngine` already supports:

- `health_delta` — partial progress from the grader
- `knowledge_delta` — one-time reward for discovering diagnostic facts (section 9's "step-level verifier")
- `action_penalty` — per-step cost to discourage idle loops

plus `anti_hack_reward` and `safety_reward` apply stepwise filters inside each
rollout, so feedback is not only final-outcome.

## 10. the right training stack

- trl `GRPOTrainer` imported in both training scripts
- unsloth `FastLanguageModel` with `load_in_4bit=True`, lora `r=16`
- openenv for the env interface (server + client) with `--env-urls` pointing
  at one or more hosted spaces for rollout parallelism

## 11. grpo / rlvr style

reward is rlvr: the grader is a deterministic file-system check, not a
learned reward model. `solve_reward` is binary, all shaping terms are
bounded, and the grader's `grade()` is pure python with no llm in the loop.

## 12. keep inference fast

- **reset latency**: **p50 2.40 ms** in copy-mode, <1 ms on fuse-overlayfs
  hosts. bench: [`bench/bench_reset.py`](./bench/bench_reset.py) via `make bench`
- unsloth 4-bit inference path enabled in both trainers (`FastLanguageModel.for_inference`)
- rollouts distributed across multiple hf spaces via `RemoteEndpointPool`
  round-robin in [`training/remote_env.py`](./training/remote_env.py)

## 13. deploy early

- live space: [`huggingmenfordays/enterprise-hpc-openenv`](https://huggingface.co/spaces/huggingmenfordays/enterprise-hpc-openenv) — public url `https://huggingmenfordays-enterprise-hpc-openenv.hf.space`
- `Dockerfile`s are already tuned for hf spaces
- [`docs/hf_spaces_deploy.md`](./docs/hf_spaces_deploy.md) covers both
  the first-time push and the **orphan-branch redeploy trick** needed
  to push over our history (xet rejects the `.venv/` + png binaries in
  the `final-round` history)
- `TODO_FOR_USER.md` section 2 has the exact copy-pasteable push recipe

## 14. scale after stable

[`Makefile`](./Makefile) encodes the guide's recommended order:

1. `make gold` — every scenario is deterministically solvable
2. `make bench` — reset latency under 3 ms
3. `make eval` — gold vs random vs bad policy leaderboard
4. `make dry` — rollout plumbing works without gpu
5. `make train` — tiny grpo run
6. `make train-remote ENV_URLS=...` — scale to multiple hosted spaces

only step 6 requires gpu + cloud credentials.

## 15. monitor the right things

[`training/logger.py`](./training/logger.py) writes per-grpo-step metrics to
`runs/<run>/<run>.metrics.jsonl` with:

- `reward_mean`, `reward_max`
- `solve_rate` (critical "function works" column called out in §15)
- `health_mean`
- `steps_mean`
- `task_mix`
- `wall_seconds`

plus transcripts are sampled every 5 steps into
`runs/<run>/transcripts/step_*.jsonl`. optional tensorboard + wandb + hf hub
uploads happen automatically when `--wandb-project` / `--hub-repo` are set.

## 16. save models correctly

both trainers accept `--save-adapter-only`. when set, only the lora adapter is
saved via `model.save_pretrained(...)` and the risky "upcast 4-bit to 16-bit
then merge" path is skipped, matching the guide's explicit warning.

```bash
python -m training.train_hpc_outage --save-adapter-only ...
python -m training.hpc_openenv_gemma --save-adapter-only --env-urls ...
```

## 17. team split

the repo naturally maps onto the guide's recommended four-person split:

- **person a (environment)**: owns [`sysadmin_env/`](./sysadmin_env/), [`hpc_gym.py`](./hpc_gym.py), [`bench/`](./bench/)
- **person b (verifier / rewards)**: owns [`sysadmin_env/rewards.py`](./sysadmin_env/rewards.py), [`training/reward_functions.py`](./training/reward_functions.py), [`tools/verify_gold_trajectory.py`](./tools/verify_gold_trajectory.py)
- **person c (training)**: owns [`training/`](./training/), [`Makefile`](./Makefile) targets
- **person d (demo / product)**: owns [`docs/pitch.md`](./docs/pitch.md), [`docs/hf_blog.md`](./docs/hf_blog.md), [`docs/video_script.md`](./docs/video_script.md)

## 18. 1-day execution plan

covered phase-by-phase in [`GETTING_STARTED.md`](./GETTING_STARTED.md).

## 19. what judges will find compelling

| compelling factor | repo evidence |
| --- | --- |
| clear environment design | nine tasks, dataclasses + fastapi, openenv standard contract |
| objective reward functions | six-component rlvr reward stack |
| evidence the model improved | `docs/assets/reward_curve_demo.png` (gpu-free) + the real grpo curve from `training/hpc_colab.ipynb` (tracked in TODO #1) |
| reward-hacking prevention | destructive command patterns, `anti_hack_reward`, grader-owned paths, transcript sampling |
| reproducible deployment | `Dockerfile`, `openenv.yaml`, hf spaces recipe |
| sharp demo | `docs/video_script.md`, `make gold && make bench && make eval && make reward-demo` |

## 20. theme directions

we target **#3.1 world modeling / professional tasks** (primary), the
**scaler ai labs multi-app rl environment for enterprise workflows** bonus
(six apps: slurm, munge, systemd, nvidia driver, nfs, apache ood), and **#2
long-horizon planning & instruction following** (8-14 step gold trajectories).

## 21. common mistakes to avoid — self-check

| mistake | how we avoid it |
| --- | --- |
| task so hard success probability is zero | `make gold` proves every scenario is solvable; curriculum flag ramps difficulty |
| using only one reward function | six independent reward functions (`training/reward_functions.py`) |
| not checking for reward hacking | `anti_hack_reward` + `safety_reward` + periodic transcript dumps |
| training before env is stable | `make gold && make bench && make eval` run without any gpu |
| relying only on average reward | logger tracks solve_rate, steps_mean, task_mix, and dumps transcripts |
| forgetting timeouts / sandbox limits | `DEFAULT_STEP_TIMEOUT`, `DEFAULT_SHELL_TIMEOUT`, `max_runtime_minutes: 20` |
| saving lora/qlora incorrectly | `--save-adapter-only` flag + warning in this doc |

## 22. learning resources checklist

we reference every primary link from the guide in [`README.md`](./README.md)
and [`docs/hf_blog.md`](./docs/hf_blog.md), including openenv core, the hf hub
org, the tutorial examples, and the mega-lecture modules.

## faq coverage highlights (1-58)

- **rlvr vs learned reward model (§4, §11, §24)**: we use rlvr; the grader is pure python
- **why rl environments matter (§5, §7 of faq, §25)**: we expose the full act/observe/act loop via fastapi, not a static dataset
- **trl + grpo (§7, §8, §25)**: `GRPOTrainer` with six reward functions
- **unsloth (§8, §59)**: `FastLanguageModel` 4-bit qlora, `for_inference(...)`
- **curriculum (§14)**: `--curriculum` flag, three-bucket unlock schedule
- **process supervision (§11)**: per-step `health_delta` + `knowledge_delta` + `safety_reward` + `anti_hack_reward`
- **goodhart / specification gaming (§38, §42)**: binary `solve_reward` primary + bounded shaping caps
- **long-horizon problems (§51)**: curriculum + 16-turn cap + `steps_mean` tracking
- **identical runs diverging (§49)**: seeds plumbed everywhere (`args.seed`, `random.randrange` rollout seed, `GRPOConfig.seed`, `FastLanguageModel.random_state`)
- **dataset staleness (§48, rlve)**: six scenarios rotated per rollout; the registry is pluggable

## unsloth recipe references

- gpt-oss 2048 game rl (§59.2): we use the same env-driven pattern — our env
  is the hpc cluster, not a 2048 board
- advanced qwen3 grpo reward shaping (§59.1): our six-way reward stack plays
  the same role
- scheduler grpo (§59.4): reward tied to output format + task correctness is
  mirrored by our `format_reward` + `solve_reward`

---

## what still requires a human

items in `TODO_FOR_USER.md`:

1. capture a real gpu grpo reward curve (colab / kaggle notebook is ready; apr 23 reward-pipeline fixes land on next `git pull`)
2. ~~deploy to hf spaces~~ ✅ live at `huggingmenfordays/enterprise-hpc-openenv`
3. record the 90-second demo video
4. submit the form

everything the guide describes at the code, reward, env, and training-loop
level is already shipped in this repo.
