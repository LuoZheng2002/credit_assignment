#!/bin/bash

#SBATCH --job-name=oneshot_judging
#SBATCH --output=slurm/logs/judging_%j.out
#SBATCH --error=slurm/logs/judging_%j.err
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --account=bfsl-delta-cpu

set -euo pipefail

NOTIFY_START_MSG=""
NOTIFY_END_MSG=""
ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --notify-start-message)
            NOTIFY_START_MSG="$2"; shift 2 ;;
        --notify-end-message)
            NOTIFY_END_MSG="$2"; shift 2 ;;
        *)
            ARGS+=("$1"); shift ;;
    esac
done

if [[ -n "$NOTIFY_END_MSG" ]]; then
    trap 'python research-utility/scripts/pushover_notify.py "$NOTIFY_END_MSG"' EXIT
fi

REPO_ROOT="$SLURM_SUBMIT_DIR"
cd "$REPO_ROOT"
export PATH="$REPO_ROOT/.venv/bin:$PATH"

if [[ -n "$NOTIFY_START_MSG" ]]; then
    python research-utility/scripts/pushover_notify.py "$NOTIFY_START_MSG"
fi

echo "Running CPU-only oneshot judging: ${ARGS[*]}"
echo "Repo root: $REPO_ROOT"

scripts/hpc/cargo_run_with_rebuild.sh bin_oneshot_judging "${ARGS[@]}"
