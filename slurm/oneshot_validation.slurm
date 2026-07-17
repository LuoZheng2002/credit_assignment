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

if [[ -n "$NOTIFY_END_MSG" ]]; then
    trap 'python research-utility/scripts/pushover_notify.py "$NOTIFY_END_MSG"' EXIT
fi

REPO_ROOT="$SLURM_SUBMIT_DIR"
cd "$REPO_ROOT"

export PATH="$REPO_ROOT/.venv/bin:$PATH"

export SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.70}"
export SGLANG_MAX_RUNNING_REQUESTS="${SGLANG_MAX_RUNNING_REQUESTS:-32}"
export VALIDATION_MAX_CONCURRENT_ROLLOUT="${VALIDATION_MAX_CONCURRENT_ROLLOUT:-32}"
export VLLM_VENV="${VLLM_VENV:-/work/nvme/bhph/zluo8/credit_assignment/venvs/vllm-0.25.1-cu129}"
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.70}"
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-4096}"
export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-32}"
export VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-4096}"

_NVIDIA_LIBS=$(find "$REPO_ROOT/.venv/lib/python3.12/site-packages/nvidia" \
    -name "lib" -type d -print 2>/dev/null | tr '\n' ':')
if [ -n "$_NVIDIA_LIBS" ]; then
    export LD_LIBRARY_PATH="${_NVIDIA_LIBS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

echo "Running oneshot validation with config: $CONFIG_PATH"
echo "Repo root: $REPO_ROOT"

if [[ -n "$NOTIFY_START_MSG" ]]; then
    python research-utility/scripts/pushover_notify.py "$NOTIFY_START_MSG"
fi

cargo run --release --bin bin_oneshot_validation -- --config-path "$CONFIG_PATH"
