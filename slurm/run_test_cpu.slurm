#!/bin/bash
#SBATCH --job-name=test_cpu
#SBATCH --output=slurm/logs/test_cpu_%j.out
#SBATCH --error=slurm/logs/test_cpu_%j.err
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --account=bfsl-delta-cpu

set -euo pipefail

if [ $# -lt 3 ]; then
    echo "Usage: sbatch --time <HH:MM:SS> slurm/run_test_cpu.slurm <testing-config-json> <rollout-secs> <phase> [job-label]" >&2
    exit 1
fi

CONFIG_PATH="$1"
ROLLOUT_SECS="$2"
PHASE="$3"
JOB_LABEL="${4:-test_cpu}"

REPO_ROOT="$SLURM_SUBMIT_DIR"
cd "$REPO_ROOT"

export PATH="$REPO_ROOT/.venv/bin:$PATH"

python3 research-utility/scripts/pushover_notify.py "Started $JOB_LABEL test $PHASE job ${SLURM_JOB_ID}"
trap 'python3 research-utility/scripts/pushover_notify.py "Finished $JOB_LABEL test $PHASE job ${SLURM_JOB_ID}"' EXIT

echo "Running bin_run_test CPU phase label=$JOB_LABEL config=$CONFIG_PATH rollout_secs=$ROLLOUT_SECS phase=$PHASE"
scripts/hpc/cargo_run_with_rebuild.sh bin_run_test \
    --testing-configs-path "$CONFIG_PATH" \
    --rollout-secs "$ROLLOUT_SECS" \
    --num-gpus 1 \
    --inference-backend vllm \
    --phase "$PHASE"
