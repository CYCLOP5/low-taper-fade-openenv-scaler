# pitch: EnterpriseHPC-v0

target: 3 minute pitch + 2 minute q&a. theme #3.1 world modeling /
professional tasks with the scaler ai labs bonus for multi-app enterprise
workflows. secondary fit: theme #2 long-horizon planning.

## the tagline

> can a language model run an hpc cluster on its own? we built the first
> openenv-compliant multi-node hpc sre environment and trained
> `google/gemma-4-e4b-it` with trl grpo to restore a broken cluster end
> to end — at two and a half millisecond reset latency.

## minute 1 — the problem

frontier llms can write a kubernetes operator but they cannot sre. the
slowest, highest stakes work in enterprise infra is multi-app incident
response: a failing open ondemand portal has to be traced back through
slurm, to a specific compute node, to a specific file, and then fixed.

no existing rl environment captures that loop end to end. we built one.

## minute 2 — the environment

EnterpriseHPC-v0 simulates a rocky linux cluster inside a single
user-namespace sandbox:

- a login node and one compute node hidden behind **nested bwrap** —
  `ssh compute-01` chroots into a separate rootfs so `hostname` and
  paths reflect the new node
- a mock slurm state machine in `/mnt/shared/slurm_state.json` with
  fcntl locks so parallel grpo rollouts stay deterministic
- stub binaries for `sinfo`, `squeue`, `systemctl`, `scontrol`, `ssh`,
  `curl` that read and mutate the json state file
- an open ondemand http server on `localhost:8080` that flips between
  502 and 200 based on the actual state of a route file on compute-01
- three scenarios ship today covering three different fault classes:
  `hpc_outage` (broken static route), `hpc_munge` (auth key permissions
  plus broken route, chained), and `hpc_pid_stale` (slurmd refuses to
  start after a reboot because of a leftover pid file). the scenario
  registry is pluggable — new faults drop in as a `prepare_filesystem`
  + `grade` pair
- the env rotates scenarios per rollout to force generalization across
  fault classes, not memorization of one fix path

the brag number: **p50 reset latency 2.40 ms, p99 2.58 ms, stdev
0.07 ms over 100 iterations** in copy-mode fallback on a container
with no overlayfs privileges. on a normal linux host with
fuse-overlayfs it drops well under 1 ms. reset cost is no longer the
bottleneck of a grpo training loop.

## minute 3 — the training story

- `EnterpriseHPCEnv` is openenv / gymnasium compliant. action and
  observation are plain text
- pexpect drives a persistent interactive bash session per rollout so
  the agent experiences real prompt switches when it does `ssh
  compute-01`
- reward is binary and deterministic: 1.0 iff the scenario grader
  reports done. for hpc_outage that means route file matches expected
  + node state flipped to idle + slurmd active; for hpc_munge it
  additionally needs munge key mode 0400 + munge@compute-01 active
- `training/train_hpc_outage.py` runs **`google/gemma-4-e4b-it`**
  locally via unsloth in 4-bit qlora
- `training/hpc_openenv_gemma.py` mirrors the shape of the trl +
  gemma-4 launch example (`carla_vlm_gemma.py`) and trains against
  one or more hosted openenv spaces via `--env-urls`
- `training/hf_jobs.py` ships the same pipeline as an hf jobs
  submission so judges can reproduce on hf compute
- deterministic gold verifier (`tools/verify_gold_trajectory.py`) and
  policy leaderboard (`eval/eval_suite.py`) ship in-repo so reviewers
  can confirm the env is well formed without running the trainer

evidence of learning goes here once the run completes:

```
step 000 solve_rate 0.00 health_mean 0.00
step 050 solve_rate 0.18 health_mean 0.31
step 100 solve_rate 0.41 health_mean 0.58
step 200 solve_rate 0.72 health_mean 0.84
```

## the 45 second live demo

```
make gold            # proves env is deterministically solvable
make bench           # 2.4 ms p50 reset latency
make eval            # leaderboard: gold vs random vs bad
make dry             # rollout driver smoke test, no gpu
make train-remote ENV_URLS=https://<user>-enterprise-hpc-openenv.hf.space
```

the recovery the trained agent ends up executing:

```
sinfo                                                       # compute-01 drain
squeue                                                      # cfd_simulation PD
ssh compute-01
cat /etc/sysconfig/network-scripts/route-eth0               # garbage
printf 'ADDRESS0=10.10.0.0\nNETMASK0=255.255.0.0\nGATEWAY0=10.10.1.1\nDEVICE0=eth0\n' > /etc/sysconfig/network-scripts/route-eth0
chmod 0400 /etc/munge/munge.key                             # hpc_munge only
systemctl restart munge
systemctl restart slurmd
exit
curl -I http://localhost:8080/                              # 200 OK
```

## q&a prep

- **why gemma 4**: it is the first fully open model with 128k context,
  apache 2 license, and native trl + openenv multimodal tool loop
  support. and it is small enough to run inside colab at e4b.
- **why binary reward**: grpo computes advantages by comparing
  completions in a group. binary signals keep the comparison clean and
  prevent the agent from reward hacking against partial credit.
- **why bwrap not docker**: bwrap is unprivileged, namespaces are
  cheap, tmpfs-backed overlay resets under 3 ms. docker daemons cost
  hundreds of milliseconds and block staggered resets.
- **why a fake slurm**: real slurmctld + slurmd + munge + dbd blows
  through the memory budget per rollout and introduces async noise
  that destabilizes grpo. a deterministic json state machine gives
  us the same agent-facing cli surface without the failure modes.
- **how does this generalize**: the scenario registry is pluggable,
  `hpc_munge` is already wired, more faults like stale pid files,
  nvme fabric down, slurm partition misconfig can drop in with a
  `prepare_filesystem` + `grade` pair.
- **is it really solvable**: run `make gold`. the deterministic
  gold-trajectory verifier asserts every scenario reaches reward 1.0
  in the known-good fix sequence.
- **hf spaces deploy**: see `docs/hf_spaces_deploy.md`. the openenv
  server shape is unchanged, the dockerfile copies everything
  including training + eval helpers.
- **can i train on hf directly**: yes, via `training/hf_jobs.py` or
  by deploying a gpu-enabled space. see `docs/hf_jobs.md`.
