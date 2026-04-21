# teaching an llm to sre: EnterpriseHPC-v0 on openenv

tl;dr we shipped an openenv compliant gymnasium environment that
simulates a 224 core rocky linux hpc cluster inside a single user
namespace sandbox, resets in **2.40 ms p50**, and trains
**google/gemma-4-e4b-it** with trl grpo to recover a broken cluster
end to end. the same training script can run locally, in colab, or
against a fleet of hf spaces via `--env-urls`.

## why

the slowest, highest stakes work in enterprise infra is multi-app
incident response. an open ondemand portal returns 502. the compute
partition is drained. there is a failing slurmd somewhere. to fix it
you navigate login -> compute-01 over ssh, inspect route configs and
munge keys, restart services in the right order, and verify via curl.
frontier llms have never trained on that loop.

EnterpriseHPC-v0 turns that loop into an rl environment.

## what is inside

- nested bwrap for lateral movement. `ssh compute-01` chroots the
  shell into a separate rootfs so `hostname` and filesystem paths
  reflect the new node
- fuse-overlayfs with upperdir and workdir on `/dev/shm` for
  microsecond copy on write. kernel overlay and a copy fallback are
  supported for hosts without fuse privileges
- a deterministic slurm state machine in
  `/mnt/shared/slurm_state.json` with fcntl locks so many parallel
  rollouts cannot corrupt each other
- python stubs for sinfo, squeue, systemctl, scontrol, curl, ssh that
  read and mutate the json state, and a lightweight open ondemand
  http server that returns 502 until the underlying fault is fixed
- three scenarios ship today and are rotated per rollout
  - `hpc_outage` compute-01 drain from a broken route-eth0
  - `hpc_munge` compute-01 drain from a munge key with wrong mode and
    a broken route (chained)
  - `hpc_pid_stale` slurmd refuses to restart after reboot because of a
    leftover `/var/run/slurmd.pid`
- the gymnasium env `EnterpriseHPC-v0` wraps it all with pexpect so
  the policy experiences real interactive bash prompts

## how fast

```
| mount | n | p50 ms | p95 ms | p99 ms | max ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| copy | 100 | 2.40 | 2.56 | 2.58 | 2.87 |
```

that is in the ci friendly copy mode. real fuse-overlayfs on a linux
host drops well under 1 ms. reset latency is no longer the grpo
bottleneck.

## training with gemma 4

local training with unsloth + 4bit qlora:

```
python -m training.train_hpc_outage \
  --model google/gemma-4-e4b-it \
  --group-size 4 --max-turns 12 \
  --num-train-steps 100 \
  --scenarios hpc_outage,hpc_munge,hpc_pid_stale
```

remote training against hosted openenv spaces (same shape as the
trl + gemma-4 + carla example from the launch post):

```
python -m training.hpc_openenv_gemma \
  --env-urls https://<user>-enterprise-hpc-openenv.hf.space \
             https://<user>-enterprise-hpc-openenv-2.hf.space \
  --model google/gemma-4-e4b-it \
  --group-size 4 --max-turns 12 --num-train-steps 200
```

submit to hf jobs:

```
python -m training.hf_jobs \
  --env-urls https://<user>-enterprise-hpc-openenv.hf.space \
  --gpu a10g-large \
  --num-train-steps 300
```

the training scripts use unsloth for 4bit qlora loading and trl
`GRPOTrainer` with a custom rollout function that drives the env one
turn at a time. the reward is binary from the deterministic task
grader, which is exactly the signal grpo wants.

a colab notebook at `training/hpc_colab.ipynb` runs both the local
and remote paths on a single t4 / l4 / a100.

## what the agent learns

before training a random policy wanders around `sinfo` and never edits
the route file. after ~100 steps of grpo the agent reliably:

1. runs `sinfo` and `squeue` to locate the drained node
2. lateral moves with `ssh compute-01`
3. inspects `/etc/sysconfig/network-scripts/route-eth0`
4. writes the correct route with `printf ... >` (no heredocs allowed)
5. for the munge variant also `chmod 0400 /etc/munge/munge.key`
6. restarts munge then slurmd in that order
7. exits back to login and verifies with `curl -I http://localhost:8080`

## prove it is solvable

before any training, reviewers can run:

```
make gold   # deterministic gold-trajectory verifier
make eval   # gold vs random vs bad policies, writes runs/eval/leaderboard.md
make bench  # reset-latency benchmark
```

## try it

- repo: https://github.com/your-org/low-taper-fade-openenv-scaler
- hf space (env server): https://huggingface.co/spaces/your-org/enterprise-hpc-openenv
- colab: `training/hpc_colab.ipynb`
- pitch doc: `docs/pitch.md`
- hf jobs guide: `docs/hf_jobs.md`
- spaces deploy: `docs/hf_spaces_deploy.md`
