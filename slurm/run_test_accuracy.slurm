#!/bin/bash
#SBATCH --job-name=test_accuracy
#SBATCH --output=slurm/logs/test_accuracy_%j.out
#SBATCH --error=slurm/logs/test_accuracy_%j.err
#SBATCH --gres=gpu:nvidia_a100:1
#SBATCH --cpus-per-task=32
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --account=bfsl-delta-gpu

set -euo pipefail

if [ $# -lt 2 ]; then
    echo "Usage: sbatch --time <HH:MM:SS> slurm/run_test_accuracy.slurm <testing-config-json> <rollout-secs> [job-label] [phase]" >&2
    exit 1
fi

CONFIG_PATH="$1"
ROLLOUT_SECS="$2"
JOB_LABEL="${3:-test_accuracy}"
PHASE="${4:-all}"

REPO_ROOT="$SLURM_SUBMIT_DIR"
cd "$REPO_ROOT"

source "$REPO_ROOT/slurm/setup_cuda_12_8.sh"

export PATH="$REPO_ROOT/.venv/bin:$PATH"
export VLLM_VENV="${VLLM_VENV:-/work/hdd/bhph/zluo8/credit_assignment/venvs/vllm-latest-cu130}"
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-4096}"
export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-32}"
export VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-4096}"
export VALIDATION_MAX_CONCURRENT_ROLLOUT="${VALIDATION_MAX_CONCURRENT_ROLLOUT:-32}"
export INFERENCE_WRAPPER_PORT="${INFERENCE_WRAPPER_PORT:-$((30000 + SLURM_JOB_ID % 1000))}"
export XDG_CACHE_HOME="${SLURM_TMPDIR:-/tmp}/credit_assignment_cache_${SLURM_JOB_ID}"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export CUDA_CACHE_PATH="$XDG_CACHE_HOME/cuda"
mkdir -p "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH"

_NVIDIA_LIBS=$(find "$REPO_ROOT/.venv/lib/python3.12/site-packages/nvidia" \
    -name "lib" -type d -print 2>/dev/null | tr '\n' ':')
if [ -n "$_NVIDIA_LIBS" ]; then
    export LD_LIBRARY_PATH="${_NVIDIA_LIBS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

python3 research-utility/scripts/pushover_notify.py "Started $JOB_LABEL test job ${SLURM_JOB_ID}"
trap 'python3 research-utility/scripts/pushover_notify.py "Finished $JOB_LABEL test job ${SLURM_JOB_ID}"' EXIT

echo "Running bin_run_test label=$JOB_LABEL config=$CONFIG_PATH rollout_secs=$ROLLOUT_SECS phase=$PHASE wrapper_port=$INFERENCE_WRAPPER_PORT"
scripts/hpc/cargo_run_with_rebuild.sh bin_run_test \
    --testing-configs-path "$CONFIG_PATH" \
    --rollout-secs "$ROLLOUT_SECS" \
    --num-gpus 1 \
    --inference-backend vllm \
    --phase "$PHASE"
