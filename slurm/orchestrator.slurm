#!/bin/bash

#SBATCH --job-name=orch
#SBATCH --output=slurm/logs/orch_%j.out
#SBATCH --error=slurm/logs/orch_%j.err
#SBATCH --gres=gpu:nvidia_a100:1
#SBATCH --cpus-per-task=32
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --account=bfsl-delta-gpu
#SBATCH --time=48:00:00

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: sbatch slurm/orchestrator.slurm <config-path> [notify-start-message] [notify-end-message]" >&2
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
export VLLM_VENV="/u/zluo8/credit_assignment/venvs/vllm-latest-cu130"
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-4096}"
export VLLM_CACHE_HOME="${VLLM_CACHE_HOME:-/u/${USER:-zluo8}/.cache/credit_assignment_vllm}"
mkdir -p "$VLLM_CACHE_HOME/xdg" "$VLLM_CACHE_HOME/triton" "$VLLM_CACHE_HOME/torchinductor" "$VLLM_CACHE_HOME/cuda"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$VLLM_CACHE_HOME/xdg}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$VLLM_CACHE_HOME/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$VLLM_CACHE_HOME/torchinductor}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-$VLLM_CACHE_HOME/cuda}"
unset VLLM_MAX_NUM_SEQS VLLM_MAX_NUM_BATCHED_TOKENS VLLM_ENFORCE_EAGER
export VLLM_INITIAL_HEALTH_TIMEOUT_SECS="${VLLM_INITIAL_HEALTH_TIMEOUT_SECS:-2700}"
export VLLM_MODEL_UPDATE_HEALTH_TIMEOUT_SECS="${VLLM_MODEL_UPDATE_HEALTH_TIMEOUT_SECS:-2700}"
export INFERENCE_WRAPPER_HEALTH_TIMEOUT_SECS="${INFERENCE_WRAPPER_HEALTH_TIMEOUT_SECS:-2700}"
export INFERENCE_UPDATE_MODEL_TIMEOUT_SECS="${INFERENCE_UPDATE_MODEL_TIMEOUT_SECS:-2700}"

echo "Running orchestrator with config: $CONFIG_PATH"
echo "Repo root: $REPO_ROOT"

if [[ -n "$NOTIFY_START_MSG" ]]; then
    python research-utility/scripts/pushover_notify.py "$NOTIFY_START_MSG"
fi

cargo run --release --bin bin_orchestrator -- --config-path "$CONFIG_PATH"
