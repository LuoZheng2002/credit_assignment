#!/bin/bash

#SBATCH --job-name=oneshot_prune
#SBATCH --output=slurm/logs/prune_%j.out
#SBATCH --error=slurm/logs/prune_%j.err
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --account=bfsl-delta-cpu

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: sbatch slurm/oneshot_prune_cpu.slurm <config-path> [notify-start-message] [notify-end-message]" >&2
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

if [[ -n "$NOTIFY_START_MSG" ]]; then
    python research-utility/scripts/pushover_notify.py "$NOTIFY_START_MSG"
fi

echo "Running oneshot prune with config: $CONFIG_PATH"
echo "Repo root: $REPO_ROOT"

scripts/hpc/cargo_run_with_rebuild.sh bin_oneshot_prune --config-path "$CONFIG_PATH"
