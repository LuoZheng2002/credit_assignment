#!/bin/bash

#SBATCH --job-name=validation
#SBATCH --output=slurm/logs/validation_%j.out
#SBATCH --error=slurm/logs/validation_%j.err
#SBATCH --cpus-per-task=32
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --account=bfsl-delta-gpu

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: scripts/hpc/bin_oneshot_validation.py --config-path <config-path>" >&2
    echo "   or: sbatch --gres=gpu:nvidia_a100:<num-gpus> slurm/oneshot_validation.slurm <config-path> [notify-start-message] [notify-end-message]" >&2
    exit 1
fi

CONFIG_PATH="$1"
NOTIFY_START_MSG="${2:-}"
NOTIFY_END_MSG="${3:-}"
EXTRA_ARGS=("${@:4}")

if [[ -n "$NOTIFY_END_MSG" ]]; then
    trap 'python research-utility/scripts/pushover_notify.py "$NOTIFY_END_MSG"' EXIT
fi

REPO_ROOT="$SLURM_SUBMIT_DIR"
cd "$REPO_ROOT"

source "$REPO_ROOT/slurm/setup_cuda_12_8.sh"

export PATH="$REPO_ROOT/.venv/bin:$PATH"

export SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.70}"
export SGLANG_MAX_RUNNING_REQUESTS="${SGLANG_MAX_RUNNING_REQUESTS:-32}"
export VALIDATION_MAX_CONCURRENT_ROLLOUT=200
export VLLM_VENV="${VLLM_VENV:-/work/hdd/bhph/zluo8/credit_assignment/venvs/vllm-latest-cu130}"
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-4096}"
export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-128}"
export VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-4096}"
export VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-1}"
export VLLM_INITIAL_HEALTH_TIMEOUT_SECS="${VLLM_INITIAL_HEALTH_TIMEOUT_SECS:-1800}"
export VLLM_MODEL_UPDATE_HEALTH_TIMEOUT_SECS="${VLLM_MODEL_UPDATE_HEALTH_TIMEOUT_SECS:-1800}"
export INFERENCE_WRAPPER_HEALTH_TIMEOUT_SECS="${INFERENCE_WRAPPER_HEALTH_TIMEOUT_SECS:-1800}"
export INFERENCE_UPDATE_MODEL_TIMEOUT_SECS="${INFERENCE_UPDATE_MODEL_TIMEOUT_SECS:-1800}"
export INFERENCE_WRAPPER_PORT="${INFERENCE_WRAPPER_PORT:-$((30000 + SLURM_JOB_ID % 1000))}"
export XDG_CACHE_HOME="${SLURM_TMPDIR:-/tmp}/credit_assignment_cache_${SLURM_JOB_ID}"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export CUDA_CACHE_PATH="$XDG_CACHE_HOME/cuda"
mkdir -p "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH"

_NVIDIA_LIBS=$(find "$REPO_ROOT/.venv/lib/python3.12/site-packages/nvidia" \
    -name "lib" -type d -print 2>/dev/null | tr '\n' ':')
if [ -n "$_NVIDIA_LIBS" ]; then
    export LD_LIBRARY_PATH="${_NVIDIA_LIBS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

echo "Running oneshot validation with config: $CONFIG_PATH"
echo "Repo root: $REPO_ROOT"
echo "Inference wrapper port: $INFERENCE_WRAPPER_PORT"

if [[ -n "$NOTIFY_START_MSG" ]]; then
    python research-utility/scripts/pushover_notify.py "$NOTIFY_START_MSG"
fi

scripts/hpc/cargo_run_with_rebuild.sh bin_oneshot_validation --config-path "$CONFIG_PATH" "${EXTRA_ARGS[@]}"
