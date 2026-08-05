#!/bin/bash

#SBATCH --job-name=training_chunk_validation
#SBATCH --output=slurm/logs/training_chunk_validation_%j.out
#SBATCH --error=slurm/logs/training_chunk_validation_%j.err
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --mem=16G

set -euo pipefail

if [ $# -lt 2 ]; then
    echo "Usage: sbatch slurm/oneshot_training_chunk_validation.slurm <config-path> <phase> [notify-start-message] [notify-end-message]" >&2
    echo "   or: sbatch slurm/oneshot_training_chunk_validation.slurm <config-path> <chunk-index> <phase> [notify-start-message] [notify-end-message]" >&2
    echo "   or: sbatch slurm/oneshot_training_chunk_validation.slurm <config-path> [notify-start-message] [notify-end-message] <phase>" >&2
    echo "   or: sbatch slurm/oneshot_training_chunk_validation.slurm <config-path> [notify-start-message] [notify-end-message] <chunk-index> <phase>" >&2
    exit 1
fi

CONFIG_PATH="$1"
CHUNK_INDEX=""
if [[ "${2:-}" =~ ^[0-9]+$ ]]; then
    CHUNK_INDEX="$2"
    PHASE="$3"
    NOTIFY_START_MSG="${4:-}"
    NOTIFY_END_MSG="${5:-}"
elif [[ "${2:-}" =~ ^(all|rollout|judge|score)$ ]]; then
    PHASE="$2"
    NOTIFY_START_MSG="${3:-}"
    NOTIFY_END_MSG="${4:-}"
elif [[ "${4:-}" =~ ^[0-9]+$ ]]; then
    NOTIFY_START_MSG="${2:-}"
    NOTIFY_END_MSG="${3:-}"
    CHUNK_INDEX="${4:-}"
    PHASE="${5:-}"
else
    NOTIFY_START_MSG="${2:-}"
    NOTIFY_END_MSG="${3:-}"
    PHASE="${4:-}"
fi

if [[ -z "$PHASE" ]]; then
    echo "Missing phase" >&2
    exit 1
fi

if [[ -n "$NOTIFY_END_MSG" ]]; then
    trap 'python research-utility/scripts/pushover_notify.py "$NOTIFY_END_MSG"' EXIT
fi

REPO_ROOT="$SLURM_SUBMIT_DIR"
cd "$REPO_ROOT"

export PATH="$REPO_ROOT/.venv/bin:$PATH"
export VLLM_VENV="${VLLM_VENV:-/work/nvme/bhph/zluo8/credit_assignment/venvs/vllm-0.25.1-cu129}"
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-4096}"
export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-32}"
export VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-4096}"
export VALIDATION_MAX_CONCURRENT_ROLLOUT=200

_NVIDIA_LIBS=$(find "$REPO_ROOT/.venv/lib/python3.12/site-packages/nvidia" \
    -name "lib" -type d -print 2>/dev/null | tr '\n' ':')
if [ -n "$_NVIDIA_LIBS" ]; then
    export LD_LIBRARY_PATH="${_NVIDIA_LIBS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

if [[ -n "$NOTIFY_START_MSG" ]]; then
    python research-utility/scripts/pushover_notify.py "$NOTIFY_START_MSG"
fi

echo "Running training chunk validation config=$CONFIG_PATH chunk=${CHUNK_INDEX:-all} phase=$PHASE"
CMD=(
    scripts/hpc/cargo_run_with_rebuild.sh bin_oneshot_training_chunk_validation
    --config-path "$CONFIG_PATH"
    --phase "$PHASE"
)
if [[ -n "$CHUNK_INDEX" ]]; then
    CMD+=(--chunk-index "$CHUNK_INDEX")
else
    CMD+=(--all-chunks)
fi
"${CMD[@]}"
