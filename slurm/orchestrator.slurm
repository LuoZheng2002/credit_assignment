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

# Prefer the main project venv for experiment executables. The inference wrapper
# resolves the vLLM runtime from VLLM_VENV when a config uses vLLM.
export PATH="$REPO_ROOT/.venv/bin:$PATH"
export VLLM_VENV="${VLLM_VENV:-/work/nvme/bhph/zluo8/credit_assignment/venvs/vllm-0.25.1-cu129}"

echo "Running orchestrator with config: $CONFIG_PATH"
echo "Repo root: $REPO_ROOT"

if [[ -n "$NOTIFY_START_MSG" ]]; then
    python research-utility/scripts/pushover_notify.py "$NOTIFY_START_MSG"
fi

cargo run --release --bin bin_orchestrator -- --config-path "$CONFIG_PATH"
