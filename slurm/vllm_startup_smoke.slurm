#!/bin/bash

#SBATCH --job-name=vllm_startup_smoke
#SBATCH --output=slurm/logs/vllm_startup_smoke_%j.out
#SBATCH --error=slurm/logs/vllm_startup_smoke_%j.err
#SBATCH --gres=gpu:nvidia_a100:1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --account=bfsl-delta-gpu

set -euo pipefail

if [ $# -lt 4 ]; then
    echo "Usage: sbatch slurm/vllm_startup_smoke.slurm <model-path> <served-name> <port> <variant-label> [extra-vllm-args...]" >&2
    exit 1
fi

MODEL_PATH="$1"
SERVED_NAME="$2"
PORT="$3"
VARIANT_LABEL="$4"
shift 4
EXTRA_ARGS=("$@")

REPO_ROOT="$SLURM_SUBMIT_DIR"
cd "$REPO_ROOT"

source "$REPO_ROOT/slurm/setup_cuda_12_8.sh"

VLLM_PYTHON="/u/zluo8/credit_assignment/venvs/vllm-latest-cu130/bin/python"
LOG_PATH="slurm/logs/vllm_startup_smoke_${SLURM_JOB_ID}_${VARIANT_LABEL}.vllm.log"
START_TS="$(date +%s)"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

echo "variant=${VARIANT_LABEL}"
echo "model=${MODEL_PATH}"
echo "port=${PORT}"
echo "start_ts=${START_TS}"
echo "extra_args=${EXTRA_ARGS[*]}"

"$VLLM_PYTHON" -m src_py.wrappers.vllm_startup_debug vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --served-model-name "$SERVED_NAME" \
    --host 127.0.0.1 \
    --port "$PORT" \
    --dtype auto \
    --skip-tokenizer-init \
    --max-model-len 4096 \
    "${EXTRA_ARGS[@]}" \
    >"$LOG_PATH" 2>&1 &
VLLM_PID=$!

cleanup() {
    if kill -0 "$VLLM_PID" 2>/dev/null; then
        kill "$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

for elapsed in $(seq 1 3600); do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "result=process_exited_before_health"
        echo "elapsed_seconds=${elapsed}"
        tail -120 "$LOG_PATH"
        exit 1
    fi
    if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
        END_TS="$(date +%s)"
        echo "result=healthy"
        echo "end_ts=${END_TS}"
        echo "startup_seconds=$((END_TS - START_TS))"
        tail -80 "$LOG_PATH"
        exit 0
    fi
    sleep 1
done

echo "result=timeout"
echo "elapsed_seconds=3600"
tail -120 "$LOG_PATH"
exit 1
