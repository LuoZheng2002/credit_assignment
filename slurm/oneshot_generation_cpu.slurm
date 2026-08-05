#!/bin/bash

#SBATCH --job-name=generation_cpu
#SBATCH --output=slurm/logs/generation_%j.out
#SBATCH --error=slurm/logs/generation_%j.err
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --account=bfsl-delta-cpu

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: scripts/hpc/bin_oneshot_generation.py --config-path <config-path>" >&2
    echo "   or: sbatch slurm/oneshot_generation_cpu.slurm <config-path> [notify-start-message] [notify-end-message]" >&2
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

if [[ -n "$NOTIFY_START_MSG" ]]; then
    python research-utility/scripts/pushover_notify.py "$NOTIFY_START_MSG"
fi

echo "Running CPU-only oneshot generation with config: $CONFIG_PATH extra_args=${EXTRA_ARGS[*]:-}"
echo "Repo root: $REPO_ROOT"

scripts/hpc/cargo_run_with_rebuild.sh bin_oneshot_generation --config-path "$CONFIG_PATH" "${EXTRA_ARGS[@]}"
