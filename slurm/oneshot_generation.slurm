#!/bin/bash

#SBATCH --job-name=generation
#SBATCH --output=slurm/logs/generation_%j.out
#SBATCH --error=slurm/logs/generation_%j.err
#SBATCH --gres=gpu:nvidia_a100:1
#SBATCH --cpus-per-task=32
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --account=bfsl-delta-gpu

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: sbatch slurm/oneshot_generation.slurm <config-path> [notify-start-message] [notify-end-message]" >&2
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

if [[ -n "$NOTIFY_START_MSG" ]]; then
    python research-utility/scripts/pushover_notify.py "$NOTIFY_START_MSG"
fi

echo "Running oneshot generation with config: $CONFIG_PATH"
echo "Repo root: $REPO_ROOT"

cargo run --release --bin bin_oneshot_generation -- --config-path "$CONFIG_PATH"
