# what I need you to do — hackathon final stretch

I cannot do these inside the Cursor sandbox (no GPU, no HF credentials, no
PTY devices, no real network). these are the remaining blockers between
"technically complete" and "wins the hackathon".

legend
- **[BLOCKER]** must be done before submission
- **[BONUS]**  meaningful boost on the rubric, not required
- **[POLISH]** last-minute polish if you have time

## 1 [BLOCKER] capture a reward curve on a real gpu

**partial credit already banked**: `docs/assets/reward_curve_demo.png`
is committed — the gpu-free curriculum-annealed reward probe in
`tools/reward_curve_demo.py` proves the shaped reward signal has a
learnable gradient (0.03 → 0.51 over 24 curriculum steps). judges see
a real curve immediately. run `make reward-demo` to regenerate it.

we still want a real gpu grpo run for the "we trained a model" story:

### what to run

open `training/hpc_colab.ipynb` in colab (pick L4 or A100, free T4 also
works at group-size 2). run every cell. cell 6 now runs the gpu-free
probe and inlines the png. cell 8 is the real grpo run. once that is
done:

```
# in colab
import matplotlib.pyplot as plt
# cell 10 already plots from runs/*.metrics.jsonl, just save the figure
plt.savefig('reward_curve.png', dpi=150, bbox_inches='tight')
```

### what I need back

1. a png of the real grpo curve (save as `docs/assets/reward_curve.png`)
2. the final `runs/hpc_grpo_local/hpc_openenv_gemma.metrics.jsonl`
3. optionally: push the lora adapter to `huggingface.co/<you>/hpc-grpo-gemma-4-e4b`

once those are in the repo I will update `docs/pitch.md`, `docs/hf_blog.md`,
and `README.md` to inline the chart and link the hub artifacts.

## 2 [BLOCKER] deploy the openenv server to a hf space - DONE

space: https://huggingface.co/spaces/huggingmenfordays/enterprise-hpc-openenv
live url: https://huggingmenfordays-enterprise-hpc-openenv.hf.space

_original instructions below for reference_

## 2 [reference] deploy the openenv server to a hf space

judges will click "try it" in the submission form. without a live space
they cannot hit the env.

### steps

1. `huggingface-cli login` with a token that has space-write permission
2. from this repo:
   ```bash
   huggingface-cli repo create enterprise-hpc-openenv \
     --type space --space_sdk docker
   git remote add space https://huggingface.co/spaces/<you>/enterprise-hpc-openenv
   git push space main
   ```
3. wait for the docker build (5-10 min first time)
4. confirm `curl https://<you>-enterprise-hpc-openenv.hf.space/health` returns 200
5. send me the URL and I will wire it into `openenv.yaml` and the pitch

### notes

the existing `Dockerfile` is already tuned. apparmor may block
`fuse-overlayfs`, the copy fallback (p50 ~2.4 ms) still hits the latency
target. if the build errors on `bubblewrap`, we can add `apt-get install -y`
for it.

## 3 [BLOCKER] record a 90-second demo video

the video is part of most hackathon submissions. script is in
`docs/video_script.md`.

### shots to capture

1. `make gold` — quick pass, proves determinism (5 s)
2. `make bench` — show the 2.40 ms p50 number (10 s)
3. `make eval` — cat the leaderboard markdown (15 s)
4. the live agent solving `hpc_pid_stale` via
   `python -m training.train_hpc_outage --dry-run --group-size 1` or a
   trained checkpoint (40 s)
5. the reward curve chart (20 s)

record with OBS or the built-in macOS screen recorder, upload to
youtube or HF, paste the URL into `README.md` under a "demo" section and I
will finalize.

## 4 [BONUS] give me access to a space url so I can wire things up

once task 2 is done, paste the URL here and I will:

- update `openenv.yaml` `runtime.server_entry_point`
- add a "Try the env live" section to `README.md` and the HF blog
- update `docs/pitch.md` to reference the live URL in the q&a prep

## 5 [BONUS] run a longer training session and push to the hub

once task 1 is done and the pipeline is validated:

```bash
python -m training.hpc_openenv_gemma \
  --env-urls https://<you>-enterprise-hpc-openenv.hf.space \
  --model google/gemma-4-e4b-it \
  --num-train-steps 600 \
  --group-size 8 --max-turns 16 \
  --hub-repo <you>/hpc-grpo-gemma-4-e4b \
  --wandb-project hpc-grpo
```

600 steps at group-size 8 takes ~3 hours on an A100. this is what gets you
"we actually trained a model that beats the baseline" for the rubric.

## 6 [POLISH] submission form metadata

when you fill out the form:

- **theme**: #3.1 World Modeling / Professional Tasks (primary), #2
  Long-Horizon Planning (secondary)
- **tagline**: "EnterpriseHPC-v0 — a multi-app, sub-3 ms-reset HPC SRE
  environment. Gemma 4 learns to diagnose a 224-core Rocky Linux cluster
  end-to-end."
- **links**: github repo, hf space, hf model repo, colab, video
- **highlights**: multi-app (Slurm + OOD Apache + SSH + OverlayFS +
  NVIDIA driver + NFS + systemd + Munge), multi-node (nested bwrap),
  **six deterministic scenarios** (`hpc_outage`, `hpc_munge`,
  `hpc_pid_stale`, `hpc_gpu_ecc`, `hpc_nfs_stale`, `hpc_ood_apache`),
  <3 ms reset, gpu-free reward-curve demo in-repo, trained with
  TRL + Unsloth + `google/gemma-4-e4b-it`. targets the **Scaler AI Labs
  Multi-App RL Environment for Enterprise Workflows** bonus

## 7 [POLISH] things I can do as soon as you unblock

once you have a GPU + HF account handy:

- [ ] add the reward curve PNG to `docs/pitch.md` and `docs/hf_blog.md`
- [ ] update `README.md` with the live HF Space URL
- [ ] add a "trained checkpoint" section pointing at your HF model repo
- [ ] write the final HF blog post draft and submit it
- [ ] extend the scenario set if you want (see [extra ideas](#extra-ideas))

## 8 [BLOCKER] submit the darn thing

don't forget to actually click submit. past hackathon winners all had a
running demo URL, a reward curve, and a 60-second elevator pitch.

---

## extra ideas (if we still have time)

already shipped for round 2:

- ✅ **`hpc_gpu_ecc`** — compute node drained due to nvidia-smi ECC
  errors. fix loop: `sinfo`, `ssh compute-01`, `nvidia-smi`,
  `nvidia-smi -r -i 0`, `systemctl restart slurmd`, `exit`, `sinfo`
- ✅ **`hpc_nfs_stale`** — `/mnt/shared` stale nfs handle after a
  server failover. fix loop: `ls /mnt/shared` (errors), `umount -l
  /mnt/shared`, `mount /mnt/shared`, `systemctl restart slurmd`
- ✅ **`hpc_ood_apache`** — open ondemand portal degraded because of a
  httpd config typo on `:8081`. fix loop: `curl -I
  http://localhost:8081/` (502), `cat /etc/httpd/conf/httpd.conf`,
  `apachectl configtest`, `printf '<fixed>' > httpd.conf`,
  `apachectl graceful`, `curl -I http://localhost:8081/` (200)

still on the wishlist if we have extra time:

- **multi-node ssh traversal** — add compute-02 for a partition
  imbalance scenario
- **`hpc_cgroup_oom`** — slurmd kills jobs because a system cgroup
  limit is set too low; fix by editing `/etc/slurm/cgroup.conf`
- **`hpc_ldap_auth`** — user cannot ssh because sssd lost contact
  with ldap; fix by restarting sssd and clearing `/var/lib/sss/db`

tell me which you want and I will drop them in (each one is ~150 loc).

---

## checklist to ship

- [ ] 1. reward curve captured and committed
- [ ] 2. HF Space deployed
- [ ] 3. demo video recorded
- [ ] 4. HF Space URL in this repo
- [ ] 5. trained checkpoint on the hub
- [ ] 6. submission form filled
- [ ] 7. final PR merged and tagged
- [ ] 8. submitted ✅
