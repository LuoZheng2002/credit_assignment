#!/bin/bash

#SBATCH --job-name=rollout
#SBATCH --output=slurm/logs/rollout_%j.out
#SBATCH --error=slurm/logs/rollout_%j.err
#SBATCH --gres=gpu:nvidia_a100:1
#SBATCH --cpus-per-task=32
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --account=bfsl-delta-gpu

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: sbatch slurm/oneshot_rollout.slurm <config-path> [notify-start-message] [notify-end-message]" >&2
    exit 1
fi

CONFIG_PATH="$1"
NOTIFY_START_MSG="${2:-}"
NOTIFY_END_MSG="${3:-}"

if [[ -n "$NOTIFY_END_MSG" ]]; then
    trap 'python research-utility/scripts/pushover_notify.py "$NOTIFY_END_MSG"' EXIT
fi

# Resolve to the repository root (sbatch copies script, so use SLURM_SUBMIT_DIR)
REPO_ROOT="$SLURM_SUBMIT_DIR"
cd "$REPO_ROOT"

source "$REPO_ROOT/slurm/setup_cuda_12_8.sh"

# Prefer the main project venv for experiment executables. The inference wrapper
# resolves the vLLM runtime from VLLM_VENV when a config uses vLLM.
export PATH="$REPO_ROOT/.venv/bin:$PATH"
export VLLM_VENV="${VLLM_VENV:-/work/nvme/bhph/zluo8/credit_assignment/venvs/vllm-0.25.1-cu129}"
export INFERENCE_WRAPPER_PORT="${INFERENCE_WRAPPER_PORT:-$((30000 + SLURM_JOB_ID % 1000))}"
export XDG_CACHE_HOME="${SLURM_TMPDIR:-/tmp}/credit_assignment_cache_${SLURM_JOB_ID}"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export CUDA_CACHE_PATH="$XDG_CACHE_HOME/cuda"
mkdir -p "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH"

echo "Running oneshot rollout with config: $CONFIG_PATH"
echo "Repo root: $REPO_ROOT"
echo "Inference wrapper port: $INFERENCE_WRAPPER_PORT"

if [[ -n "$NOTIFY_START_MSG" ]]; then
    python research-utility/scripts/pushover_notify.py "$NOTIFY_START_MSG"
fi

scripts/hpc/cargo_run_with_rebuild.sh bin_oneshot_rollout --config-path "$CONFIG_PATH"
