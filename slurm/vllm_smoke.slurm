#!/bin/bash

#SBATCH --job-name=vllm-smoke
#SBATCH --output=slurm/logs/vllm_smoke_%j.out
#SBATCH --error=slurm/logs/vllm_smoke_%j.err
#SBATCH --gres=gpu:nvidia_a100:1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --mem=48G
#SBATCH --account=bfsl-delta-gpu
#SBATCH --time=01:00:00

set -euo pipefail

REPO_ROOT="$SLURM_SUBMIT_DIR"
cd "$REPO_ROOT"

VLLM_VENV="${VLLM_VENV:-/work/nvme/bhph/zluo8/credit_assignment/venvs/vllm-0.25.1-cu129}"
BASE_PYTHON="${BASE_PYTHON:-$REPO_ROOT/.venv/bin/python}"
VLLM_WHEEL_URL="${VLLM_WHEEL_URL:-https://github.com/vllm-project/vllm/releases/download/v0.25.1/vllm-0.25.1%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl}"

echo "Repo root: $REPO_ROOT"
echo "vLLM venv: $VLLM_VENV"
echo "Base python: $BASE_PYTHON"
echo "vLLM wheel: $VLLM_WHEEL_URL"

export UV_CACHE_DIR="${UV_CACHE_DIR:-/work/nvme/bhph/zluo8/credit_assignment/uv-cache}"

mkdir -p "$(dirname "$VLLM_VENV")"
if [ ! -x "$VLLM_VENV/bin/python" ]; then
    uv venv --python "$BASE_PYTHON" "$VLLM_VENV"
fi

uv pip install --python "$VLLM_VENV/bin/python" --upgrade \
    "$VLLM_WHEEL_URL" \
    --extra-index-url https://download.pytorch.org/whl/cu129 \
    --index-strategy unsafe-best-match
uv pip show --python "$VLLM_VENV/bin/python" vllm | sed -n '1,20p'

export HF_HOME="${HF_HOME:-/work/nvme/bhph/zluo8/hf_home}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

"$VLLM_VENV/bin/python" scripts/hpc/vllm_token_smoke.py
