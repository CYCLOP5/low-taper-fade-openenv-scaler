#!/bin/bash
# sbatch recipe for the spit.ac.in gpu cluster (2x rtx 6000 ada)
#
# submits a long grpo training run for enterprise-hpc-v0. launches the openenv
# server on localhost:8000 inside the same job, then trains gemma-4-e4b-it
# against it with trl + unsloth over 2 gpus.
#
# usage: sbatch scripts/train_spit.sbatch
#
#SBATCH --job-name=hpc-grpo
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --gres=gpu:2
#SBATCH --time=12:00:00
#SBATCH --output=runs/slurm/%x-%j.out
#SBATCH --error=runs/slurm/%x-%j.err

set -euo pipefail
mkdir -p runs/slurm

echo "=== node $(hostname) job $SLURM_JOB_ID gpus $CUDA_VISIBLE_DEVICES ==="
nvidia-smi

# 1) environment setup (module + conda-forge bubblewrap, no sudo)
module load miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"

if ! conda env list | grep -q '^hpc '; then
  conda create -n hpc python=3.11 -y
  conda activate hpc
  conda install -c conda-forge bubblewrap fuse-overlayfs -y
  pip install --upgrade pip setuptools wheel
  pip install -e '.[dev,train]'
  pip install --no-deps 'unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git'
else
  conda activate hpc
fi

# 2) sanity check bwrap works in this user namespace
if ! bwrap --unshare-user -- /bin/true; then
  echo "!!! bubblewrap user namespaces unavailable on this node"
  echo "!!! fall back to running the env inside apptainer or on a hf space"
  exit 1
fi

# 3) launch the openenv server in the background on localhost
python -m server.app --host 127.0.0.1 --port 8000 > runs/slurm/env_server.log 2>&1 &
ENV_PID=$!
trap "kill $ENV_PID 2>/dev/null || true" EXIT

# wait for /health to go green
for attempt in $(seq 1 30); do
  if curl -fsS http://127.0.0.1:8000/health >/dev/null 2>&1; then
    echo "env server ready after ${attempt}s"
    break
  fi
  sleep 1
done

# 4) run grpo training. 2x48GB fits gemma-4-e4b-it in bf16 with group=8, but
#    we keep 4-bit qlora + group=8 here to leave room for longer contexts and
#    faster iteration. bump group-size to 16 if you want to saturate both gpus.
export WANDB_PROJECT=${WANDB_PROJECT:-hpc-grpo}
export HF_HUB_REPO=${HF_HUB_REPO:-}

# tunables. override per-submission via `sbatch --export=ALL,NUM_STEPS=50,...`
NUM_STEPS=${NUM_STEPS:-600}
GROUP_SIZE=${GROUP_SIZE:-8}
MAX_TURNS=${MAX_TURNS:-16}
LORA_R=${LORA_R:-32}
LORA_ALPHA=${LORA_ALPHA:-64}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-384}
MODEL=${MODEL:-google/gemma-4-e4b-it}

python -m training.hpc_openenv_gemma \
  --env-urls http://127.0.0.1:8000 \
  --model "$MODEL" \
  --group-size "$GROUP_SIZE" \
  --max-turns "$MAX_TURNS" \
  --num-train-steps "$NUM_STEPS" \
  --learning-rate 1e-5 \
  --lora-r "$LORA_R" \
  --lora-alpha "$LORA_ALPHA" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --curriculum \
  --save-adapter-only \
  --report-to tensorboard \
  --output-dir "runs/hpc_spit_${SLURM_JOB_ID}"

echo "=== training done, metrics at runs/hpc_spit_${SLURM_JOB_ID} ==="
