#!/bin/bash
#SBATCH --job-name=tree_judge_score
#SBATCH --output=slurm/logs/tree_judge_score_%j.out
#SBATCH --error=slurm/logs/tree_judge_score_%j.err
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --account=bfsl-delta-cpu

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: sbatch --time <HH:MM:SS> slurm/tree_judge_score.slurm <bin_tree_judge_score args...>" >&2
    exit 1
fi

REPO_ROOT="$SLURM_SUBMIT_DIR"
cd "$REPO_ROOT"

export PATH="$REPO_ROOT/.venv/bin:$PATH"

python3 research-utility/scripts/pushover_notify.py "Started tree_judge_score job ${SLURM_JOB_ID}"
trap 'python3 research-utility/scripts/pushover_notify.py "Finished tree_judge_score job ${SLURM_JOB_ID}"' EXIT

echo "Running bin_tree_judge_score $*"
scripts/hpc/cargo_run_with_rebuild.sh bin_tree_judge_score "$@"
