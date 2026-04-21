# training EnterpriseHPC-v0 on hugging face

three supported HF training paths. pick whichever matches your budget.

| path | gpu | setup time | best for |
| --- | --- | --- | --- |
| hf spaces gpu (persistent) | t4 / a10g | < 5 min | iterative debugging with a live environment |
| hf jobs (`training/hf_jobs.py`) | a10g / a100 / h100 | instant | big single runs you can leave unattended |
| colab / colab pro | t4 / l4 / a100 | < 2 min | demo + first training run |

all three invoke the same training entrypoints so logs and checkpoints are
interchangeable.

## 1. deploy the openenv server to a space

see `docs/hf_spaces_deploy.md` for the end-to-end guide. once deployed, your
space exposes the openenv sysadmin protocol at:

```
https://<user>-enterprise-hpc-openenv.hf.space
```

smoke test with the shipping client:

```bash
python -c "
from client import SysadminEnvClient
c = SysadminEnvClient('https://<user>-enterprise-hpc-openenv.hf.space')
ep = c.start_episode(task_id='hpc_outage')
print(ep.episode_id)
"
```

run two or three spaces in parallel for throughput. the remote env pool
round robins across them automatically.

## 2. run the training from any machine against the hosted env

this mirrors the trl+gemma-4+openenv example from the gemma 4 launch post
(`examples/scripts/openenv/carla_vlm_gemma.py`). identical shape, different
env:

```bash
python -m training.hpc_openenv_gemma \
    --env-urls https://<user>-enterprise-hpc-openenv.hf.space \
               https://<user>-enterprise-hpc-openenv-2.hf.space \
    --model google/gemma-4-e4b-it \
    --group-size 4 --max-turns 12 --num-train-steps 200 \
    --scenarios hpc_outage,hpc_munge,hpc_pid_stale \
    --hub-repo <user>/hpc-grpo-runs \
    --report-to tensorboard
```

`training/hpc_openenv_gemma.py` handles model loading with unsloth first
and falls back to plain transformers if unsloth is not available.

## 3. submit to hf jobs (fully managed, gpu-on-demand)

```bash
python -m training.hf_jobs \
    --env-urls https://<user>-enterprise-hpc-openenv.hf.space \
    --repo-url https://huggingface.co/spaces/<user>/enterprise-hpc-openenv \
    --gpu a10g-large \
    --num-train-steps 300 \
    --hub-repo <user>/hpc-grpo-runs \
    --wandb-project hpc-grpo
```

set `HF_TOKEN` and optionally `WANDB_API_KEY` in your shell. the script
uses `huggingface_hub.run_uv` if available and prints a ready-to-paste
shell script otherwise.

## 4. launching from a space with gpu

for the notebook-first workflow, create a second space with `sdk: docker`
and a gpu attached, set the startup command to

```
python -m training.hpc_openenv_gemma \
  --env-urls ${ENV_URLS} \
  --model google/gemma-4-e4b-it \
  --num-train-steps ${NUM_STEPS:-200}
```

pass `ENV_URLS` and `NUM_STEPS` via space secrets. logs stream to the
space's live logs panel and checkpoints can be pushed to a dataset repo
with `--hub-repo`.

## 5. expected artifacts

every run emits the same canonical artifacts:

- `runs/<name>/<name>.metrics.jsonl` — one jsonl line per grpo step with
  solve_rate, reward_mean, reward_max, health_mean, steps_mean, task_mix
- tensorboard event files under the output dir
- optional wandb run if `--wandb-project` is set
- optional dataset upload to `--hub-repo` for reproducible leaderboards

use these as the "showing improvement in rewards" evidence for the pitch.
