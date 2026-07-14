#!/bin/bash

set -euo pipefail

#SBATCH --job-name=rollout
#SBATCH --output=slurm/logs/rollout_%j.out
#SBATCH --error=slurm/logs/rollout_%j.err
#SBATCH --gres=gpu:nvidia_a100:1
#SBATCH --cpus-per-task=32
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --account=bfdz-delta-gpu
#SBATCH --time=48:00:00

if [ $# -lt 1 ]; then
    echo "Usage: sbatch slurm/oneshot_rollout.slurm <config-path> [notify-message]" >&2
    exit 1
fi

CONFIG_PATH="$1"
NOTIFY_MSG="${2:-}"

if [[ -n "$NOTIFY_MSG" ]]; then
    trap 'python research-utility/scripts/pushover_notify.py "$NOTIFY_MSG"' EXIT
fi

# Resolve to the repository root (sbatch copies script, so use SLURM_SUBMIT_DIR)
REPO_ROOT="$SLURM_SUBMIT_DIR"
cd "$REPO_ROOT"

# Ensure venv bin dirs are on PATH (needed for ninja, etc.)
export PATH="$REPO_ROOT/pyprojects/sglang/.venv/bin:$PATH"

echo "Running oneshot rollout with config: $CONFIG_PATH"
echo "Repo root: $REPO_ROOT"

cargo run --release --bin bin_oneshot_rollout -- --config-path "$CONFIG_PATH"
