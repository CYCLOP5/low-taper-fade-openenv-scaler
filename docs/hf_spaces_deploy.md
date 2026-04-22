# deploying EnterpriseHPC-v0 to hugging face spaces

this guide walks through hosting the openenv server on a hugging face
space so a remote agent can hit the environment over http. the space uses
the existing `Dockerfile` at the repo root.

## prerequisites

- a hugging face account
- the hub cli installed locally: `pip install huggingface_hub`
- `hf auth login` with a token that has write access to spaces

## 1 create the space

```
huggingface-cli repo create enterprise-hpc-openenv --type space --space_sdk docker
```

alternative: create it manually at
https://huggingface.co/new-space with sdk set to docker and
visibility public.

## 2 push the repo

```
git remote add space https://huggingface.co/spaces/<your-user>/enterprise-hpc-openenv
git push space main
```

the space will pick up `Dockerfile` automatically. the build takes a
few minutes because `pip install .` pulls the full dependency tree on
python 3.13. you do not need `app.py`; the `CMD` at the bottom of the
Dockerfile starts the openenv server on `:8000`.

### 2.1 redeploying a dirty / history-heavy repo (orphan-branch trick)

hugging face xet rejects pushes whose git history contains binary
blobs that were never tracked via lfs / xet (old `.venv/` artifacts,
`docs/assets/*.png`, etc). if `git push space final-round:main` fails
with:

```
! [remote rejected] final-round -> main (pre-receive hook declined)
Your push was rejected because it contains binary files.
```

the fix is to force-push a clean history-less orphan branch:

```bash
# 1 make sure you're logged in with a write token
hf auth login

# 2 remote should point at the space's git endpoint
git remote set-url space https://huggingface.co/spaces/<your-user>/enterprise-hpc-openenv

# 3 carve out a fresh orphan branch with zero history
git checkout --orphan space-deploy
git rm -rf --cached .
# keep source + docs, drop any png/binary that would blow up xet again
rm -f docs/assets/reward_curve_demo.png

# 4 stage everything still tracked and commit
git add -A
git commit -m "deploy: clean snapshot for hf space"

# 5 force-push the orphan to the space's main branch
git push space space-deploy:main --force

# 6 restore your working branch and nuke the temp branch
git checkout final-round
git branch -D space-deploy
git checkout HEAD -- docs/assets/reward_curve_demo.png
```

after the force push the space rebuilds from a one-commit history and
the binary-rejection disappears. you still develop on `final-round`
normally; only the space's `main` is rewritten.

> **live url**: https://huggingmenfordays-enterprise-hpc-openenv.hf.space
> (`huggingmenfordays/enterprise-hpc-openenv`)

## 3 expose the port correctly

spaces proxy everything to `:7860` by default. override with a space
level secret or env var:

```
PORT=7860
```

and adjust the Dockerfile `CMD` to read `$PORT` or override with a
space setting. or simpler, change the last line of the Dockerfile to:

```
CMD ["sh", "-c", "server --host 0.0.0.0 --port ${PORT:-7860}"]
```

## 4 user namespaces on spaces

hugging face spaces containers run as an unprivileged user. bwrap with
`--unshare-user` works out of the box. fuse-overlayfs does not. the
`OverlayFSManager` already handles this: kernel overlay is tried
first, fuse-overlayfs second, and a copy fallback last. expect the
copy fallback on spaces, which benches at ~3 ms reset latency, still
well within the sub 10 ms budget.

## 5 smoke test from your laptop

the minimal openenv client lives in `client.py`. hit the space with:

```
python - <<'PY'
from client import ClientError, SysadminEnvClient
c = SysadminEnvClient("https://<your-user>-enterprise-hpc-openenv.hf.space")
ep = c.start_episode(task_id="hpc_outage")
print("episode", ep.episode_id, "max_steps", ep.max_steps)
out = c.run_command(ep.episode_id, "sinfo")
print(out.stdout)
PY
```

expected first response includes `compute-01   drain   IB fabric fault`.

## 6 point the gym wrapper at the space

the `EnterpriseHPCEnv` gym wrapper talks to the sandbox via local
pexpect, not over http. for a spaces deployment, clients should use
the openenv rest api exposed by `server/` via `SysadminEnvClient`.
treat the space as the environment provider and run the training
loop anywhere with network access.

`training/remote_env.py` (`HttpEnterpriseHPCEnv`) is the thin
`RemoteEnterpriseHPCEnv` that forwards `reset` and `step` calls to
the http api, and pools multiple spaces via `RemoteEndpointPool` for
parallel rollouts. as of apr 23 2026 the server supports **per-episode
sessions** keyed on `episode_id`, so multiple concurrent rollouts
against a single space no longer clobber each other's state — the
client forwards the `episode_id` it received from `/reset` on every
subsequent `/step`, and observations now carry `grader_health`,
`grader_details`, and `ood_http_code` so the rollout driver can
compute `progress_reward` without running the grader a second time.

## 7 troubleshooting

- space fails to build on fuse-overlayfs apt install: remove the
  `fuse-overlayfs` line from the Dockerfile. the env will still work
  via kernel overlay or copy fallback
- pexpect errors about pty devices: the gym wrapper is only exercised
  inside the openenv container so this is usually not triggered from
  the space itself. it shows up when running `hpc_gym.main()` directly
  and is a signal the container was not allocated enough pty slots

## 8 what a winning submission looks like

- openenv server running on a space with a public url
- mini blog on hf with the architecture diagram and reward curve,
  linking to `docs/hf_blog.md` as the source
- colab notebook link that reproduces a training run in under an hour
- video under two minutes on youtube or linkedin with the script from
  `docs/video_script.md`
- pitch doc `docs/pitch.md` as the presentation backbone
