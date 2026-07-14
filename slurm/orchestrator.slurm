#!/bin/bash

set -euo pipefail

#SBATCH --job-name=orch
#SBATCH --output=slurm/logs/orch_%j.out
#SBATCH --error=slurm/logs/orch_%j.err
#SBATCH --gres=gpu:nvidia_a100:1
#SBATCH --cpus-per-task=32
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --account=bfdz-delta-gpu
#SBATCH --time=48:00:00

if [ $# -lt 1 ]; then
    echo "Usage: sbatch slurm/orchestrator.slurm <config-path> [notify-message]" >&2
    exit 1
fi

CONFIG_PATH="$1"
NOTIFY_MSG="${2:-}"

if [[ -n "$NOTIFY_MSG" ]]; then
    trap 'python research-utility/scripts/pushover_notify.py "$NOTIFY_MSG"' EXIT
fi

# Resolve to the repository root (this script lives in slurm/)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

echo "Running orchestrator with config: $CONFIG_PATH"
echo "Repo root: $REPO_ROOT"

cargo run --release --bin bin_orchestrator -- --config-path "$CONFIG_PATH"
