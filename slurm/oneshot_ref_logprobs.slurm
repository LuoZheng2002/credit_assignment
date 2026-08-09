#!/bin/bash

#SBATCH --job-name=ref_logprobs
#SBATCH --output=slurm/logs/ref_logprobs_%j.out
#SBATCH --error=slurm/logs/ref_logprobs_%j.err
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --account=bfsl-delta-gpu

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: scripts/hpc/bin_oneshot_ref_logprobs.py --config-path <config-path>" >&2
    echo "   or: sbatch --gres=gpu:nvidia_a100:<num-gpus> slurm/oneshot_ref_logprobs.slurm <config-path> [notify-start-message] [notify-end-message] [extra args...]" >&2
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

export PATH="$REPO_ROOT/.venv/bin:$PATH"

_NVIDIA_LIBS=$(find "$REPO_ROOT/.venv/lib/python3.12/site-packages/nvidia" \
    -name "lib" -type d -print 2>/dev/null | tr '\n' ':')
if [ -n "$_NVIDIA_LIBS" ]; then
    export LD_LIBRARY_PATH="${_NVIDIA_LIBS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

if [[ -n "$NOTIFY_START_MSG" ]]; then
    python research-utility/scripts/pushover_notify.py "$NOTIFY_START_MSG"
fi

echo "Running oneshot reference-logprob annotation with config: $CONFIG_PATH extra_args=${EXTRA_ARGS[*]:-}"
echo "Repo root: $REPO_ROOT"

scripts/hpc/cargo_run_with_rebuild.sh bin_oneshot_ref_logprobs --config-path "$CONFIG_PATH" "${EXTRA_ARGS[@]}"
