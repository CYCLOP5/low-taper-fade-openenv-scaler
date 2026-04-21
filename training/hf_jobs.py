from __future__ import annotations

import argparse
import os
import shlex
import sys


HF_JOB_SCRIPT = """\
#!/usr/bin/env -S bash -eux
git clone {repo_url} /workspace/repo
cd /workspace/repo
uv pip install -e .
uv pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" --no-deps
uv pip install 'trl>=0.12.0' datasets accelerate peft bitsandbytes tensorboard wandb huggingface_hub

python -m training.hpc_openenv_gemma \\
    --env-urls {env_urls} \\
    --model {model} \\
    --group-size {group_size} \\
    --max-turns {max_turns} \\
    --num-train-steps {num_train_steps} \\
    --scenarios {scenarios} \\
    --output-dir /workspace/runs/hpc_grpo \\
    --hub-repo "{hub_repo}" \\
    {extra_flags}
"""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env-urls", nargs="+", required=True)
    parser.add_argument("--repo-url", default="https://huggingface.co/spaces/your-user/enterprise-hpc-openenv")
    parser.add_argument("--model", default="google/gemma-4-e4b-it")
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--max-turns", type=int, default=16)
    parser.add_argument("--num-train-steps", type=int, default=300)
    parser.add_argument("--scenarios", default="hpc_outage,hpc_munge,hpc_pid_stale")
    parser.add_argument("--gpu", default="a10g-large", help="hf jobs gpu flavor")
    parser.add_argument("--hub-repo", default=os.environ.get("HF_HUB_REPO", ""))
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT"))
    parser.add_argument("--extra-flag", action="append", default=[])
    parser.add_argument("--dry-run-local", action="store_true", help="just print the command without submitting")
    return parser.parse_args()


def _render_script(args: argparse.Namespace) -> str:
    extras = " ".join(args.extra_flag)
    if args.wandb_project:
        extras = f"--wandb-project {shlex.quote(args.wandb_project)} " + extras
    return HF_JOB_SCRIPT.format(
        repo_url=args.repo_url,
        env_urls=" ".join(shlex.quote(u) for u in args.env_urls),
        model=shlex.quote(args.model),
        group_size=args.group_size,
        max_turns=args.max_turns,
        num_train_steps=args.num_train_steps,
        scenarios=shlex.quote(args.scenarios),
        hub_repo=args.hub_repo,
        extra_flags=extras,
    )


def _try_submit(args: argparse.Namespace, script: str) -> bool:
    try:
        from huggingface_hub import run_uv  # type: ignore
    except ImportError:
        try:
            from huggingface_hub import run_job  # type: ignore

            run_uv = run_job  # noqa: N806
        except ImportError:
            return False

    env = {}
    if token := os.environ.get("HF_TOKEN"):
        env["HF_TOKEN"] = token
    if project := args.wandb_project:
        env["WANDB_PROJECT"] = project
    if key := os.environ.get("WANDB_API_KEY"):
        env["WANDB_API_KEY"] = key
    try:
        job = run_uv(  # type: ignore[misc]
            script=script,
            gpu=args.gpu,
            environment=env,
        )
    except TypeError:
        job = run_uv(script=script)  # type: ignore[misc]
    print(f"hf_job submitted id {getattr(job, 'id', job)}")
    return True


def main() -> int:
    args = _parse_args()
    script = _render_script(args)
    if args.dry_run_local:
        print("# hf jobs script preview")
        print(script)
        return 0
    if not _try_submit(args, script):
        print(
            "huggingface_hub.run_uv unavailable in this install. paste the script below into\n"
            "the hf jobs web ui or upgrade `huggingface_hub` and rerun.\n",
            file=sys.stderr,
        )
        print(script)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
