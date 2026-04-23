# getting started — EnterpriseHPC-v0

end-to-end setup guide. covers a fresh linux machine, colab, and hugging
face spaces. pick the path that matches your situation.

## tl;dr fastest possible path

```bash
git clone https://github.com/<your-user>/low-taper-fade-openenv-scaler.git
cd low-taper-fade-openenv-scaler
python3.13 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -e '.[dev]'
make gold         # deterministic proof all 6 scenarios are solvable
make bench        # reset-latency benchmark (<3 ms p50 in copy mode)
make eval         # gold vs random vs bad policies, writes runs/eval/leaderboard.md
make reward-demo  # gpu-free reward-curve png, proves reward improvement
make dry          # training rollout smoke test, no gpu required
```

if everything passes, skip to [training paths](#training-paths).

## 1 prerequisites

### system packages (linux)

these are only required for the local sandbox. colab and hf jobs handle
them automatically.

```bash
sudo apt update
sudo apt install -y bubblewrap fuse-overlayfs fuse3 tini coreutils
bwrap --version           # >= 0.6 recommended
fuse-overlayfs --version  # optional, copy fallback also works
```

- `bubblewrap` (the `bwrap` binary) provides the user namespace sandbox
- `fuse-overlayfs` gives you sub-1 ms resets. missing it is fine, we fall
  back to a shutil-copy path that still hits ~2.4 ms p50

### python

- python `>=3.12` is required. python `3.13` is the current unsloth
  default (per their install docs) and the one used in `Dockerfile` +
  `server/Dockerfile`
- `pip install -e '.[dev]'` installs the package in dev mode plus all
  runtime deps (fastapi, uvicorn, gymnasium, pexpect, httpx,
  matplotlib, numpy, etc.) and pytest
- `pip install -e '.[train]'` adds the gpu-training deps (torch,
  transformers, trl, accelerate, peft, bitsandbytes, tensorboard,
  datasets). only needed on the training host

## 2 sanity checks (no gpu, 15 seconds)

run these in order. any failure means the environment is misconfigured.

```bash
# proves every scenario is deterministically solvable
python -m tools.verify_gold_trajectory -v

# measures reset latency — should be under 10 ms
python -m bench.bench_reset -n 100

# runs gold/random/bad policies against every scenario,
# writes runs/eval/leaderboard.md
python -m eval.eval_suite --trials 2
```

## 3 run the openenv server locally

```bash
make serve                 # runs the server console script on 0.0.0.0:8000
# or equivalently (after pip install -e .)
server --host 0.0.0.0 --port 8000
```

smoke test in another terminal:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/tasks
curl -X POST http://127.0.0.1:8000/reset -H 'content-type: application/json' \
  -d '{"task_id": "hpc_outage"}'
curl -X POST http://127.0.0.1:8000/step -H 'content-type: application/json' \
  -d '{"action": {"command": "sinfo"}}'
```

## 4 deploy to hugging face spaces (for remote training)

this is required if you want to train via `--env-urls https://...`. the
reference deployment lives at
[`huggingmenfordays/enterprise-hpc-openenv`](https://huggingface.co/spaces/huggingmenfordays/enterprise-hpc-openenv)
(public url: `https://huggingmenfordays-enterprise-hpc-openenv.hf.space`).

### first-time push

1. create a new space on huggingface.co — type `Docker`, any hardware tier
2. push this repo to the space:
   ```bash
   hf auth login           # once
   huggingface-cli repo create enterprise-hpc-openenv --type space --space_sdk docker
   git remote add space https://huggingface.co/spaces/<user>/enterprise-hpc-openenv
   git push space main
   ```
3. wait for the build. the space should expose your env at
   `https://<user>-enterprise-hpc-openenv.hf.space`
4. smoke test:
   ```bash
   curl https://<user>-enterprise-hpc-openenv.hf.space/health
   ```

### redeploying updates (orphan-branch trick)

this repo has `.venv/` and `docs/assets/*.png` binaries sitting in git
history that hf xet refuses to accept. a plain
`git push space final-round:main` will be rejected with
`pre-receive hook declined`. force-push a clean orphan snapshot instead:

```bash
hf auth login                                                                  # ensure token is live
git remote set-url space https://huggingface.co/spaces/<user>/enterprise-hpc-openenv

git checkout --orphan space-deploy
git rm -rf --cached .
rm -f docs/assets/reward_curve_demo.png                                        # drop binaries hf xet trips on
git add -A
git commit -m "deploy: clean snapshot for hf space"
git push space space-deploy:main --force

git checkout final-round
git branch -D space-deploy
git checkout HEAD -- docs/assets/reward_curve_demo.png                         # restore the png locally
```

your local `final-round` history stays intact; only the space's `main`
is rewritten. the build takes 5-10 min; hit `/health` to confirm it
came up green.

full guide: [`docs/hf_spaces_deploy.md`](./docs/hf_spaces_deploy.md)

## 5 training paths

### path A — local gpu (colab / single workstation)

```bash
python -m training.train_hpc_outage \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --scenarios hpc_outage,hpc_munge,hpc_pid_stale,hpc_gpu_ecc,hpc_nfs_stale,hpc_ood_apache \
  --group-size 4 --max-turns 12 --num-train-steps 100 \
  --output-dir ./runs/hpc_grpo_local
```

on colab open [`training/hpc_colab.ipynb`](./training/hpc_colab.ipynb) —
it handles all the setup. the t4 free tier works at `--group-size 2`,
l4 / a100 can push `--group-size 4+`.

### path B — remote hosted openenv (multiple spaces = throughput)

```bash
python -m training.hpc_openenv_gemma \
  --env-urls https://<user>-enterprise-hpc-openenv.hf.space \
             https://<user>-enterprise-hpc-openenv-2.hf.space \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --group-size 4 --max-turns 24 --num-train-steps 200 \
  --curriculum --save-adapter-only
```

the pool round-robins across every `--env-urls` entry for parallel
rollouts. as of apr 23 2026 the remote server supports per-episode
sessions (keyed on `episode_id`), so `group_size > 1` against a single
space no longer clobbers episode state. the default `--max-turns` is
now `24` — many scenarios need 10+ turns once format compliance and
diagnostic steps are accounted for.

### path C — hf jobs (fully managed, gpu-on-demand)

```bash
python -m training.hf_jobs \
  --env-urls https://<user>-enterprise-hpc-openenv.hf.space \
  --repo-url https://huggingface.co/spaces/<user>/enterprise-hpc-openenv \
  --gpu a10g-large \
  --num-train-steps 300 \
  --hub-repo <user>/hpc-grpo-runs
```

see [`docs/hf_jobs.md`](./docs/hf_jobs.md) for the full guide.

## 6 expected artifacts

every training run produces:

- `runs/<name>/<name>.metrics.jsonl` — reward curve time series
- tensorboard event files — `tensorboard --logdir ./runs`
- optional wandb run if `--wandb-project` is set
- optional lora adapter weights in `runs/<name>/`

to plot the reward curve locally:

```bash
tensorboard --logdir ./runs
# or use the plot cell at the bottom of training/hpc_colab.ipynb
```

## 7 troubleshooting

| symptom | fix |
| --- | --- |
| `bwrap: setting up uid map: Permission denied` | enable unprivileged user namespaces: `sudo sysctl -w kernel.unprivileged_userns_clone=1` |
| `fuse-overlayfs: not found` | harmless, we fall back to copy mode. apt install it for <1 ms resets |
| `OSError: out of pty devices` | pexpect cannot allocate a PTY. rerun on a host with `/dev/ptmx` accessible (colab, hf spaces, most linux hosts) |
| `ModuleNotFoundError: gymnasium` / `pexpect` | `pip install -e .` again, or `pip install gymnasium pexpect httpx` |
| HF Space deploy: build fails on `fuse-overlayfs` install | ignore — Spaces have apparmor restrictions, the copy fallback still works |
| `huggingface_hub.run_uv` missing | upgrade: `pip install -U huggingface_hub`. otherwise `--dry-run-local` prints the shell script |
| training OOM on T4 | lower `--group-size 2 --max-new-tokens 256`, or switch to `Qwen/Qwen2.5-Coder-3B-Instruct` / `unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit` |
| "no pty devices" when running training locally in a container | run on a linux host directly, or in colab |

## 8 one-line reproduction for judges

```bash
make help                                         # list all targets
make gold                                         # prove solvable
make bench                                        # reset latency
make eval                                         # policy leaderboard
make dry                                          # training plumbing smoke test
make train                                        # local grpo training
make train-remote ENV_URLS=https://your.hf.space  # remote openenv training
```
