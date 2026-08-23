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

source "$REPO_ROOT/slurm/setup_cuda_12_8.sh"

VLLM_VENV="${VLLM_VENV:-/u/zluo8/credit_assignment/venvs/vllm-latest-cu130}"
BASE_PYTHON="${BASE_PYTHON:-$REPO_ROOT/.venv/bin/python}"
VLLM_INSTALL_PACKAGE="${VLLM_INSTALL_PACKAGE:-vllm}"
VLLM_TORCH_BACKEND="${VLLM_TORCH_BACKEND:-cu130}"

echo "Repo root: $REPO_ROOT"
echo "vLLM venv: $VLLM_VENV"
echo "Base python: $BASE_PYTHON"
echo "vLLM install package: $VLLM_INSTALL_PACKAGE"
echo "vLLM torch backend: $VLLM_TORCH_BACKEND"

export UV_CACHE_DIR="${UV_CACHE_DIR:-/work/nvme/bhph/zluo8/credit_assignment/uv-cache}"

mkdir -p "$(dirname "$VLLM_VENV")"
if [ ! -x "$VLLM_VENV/bin/python" ]; then
    uv venv --python "$BASE_PYTHON" "$VLLM_VENV"
fi

if [ "${VLLM_SMOKE_INSTALL:-0}" = "1" ]; then
    uv pip install --python "$VLLM_VENV/bin/python" --upgrade \
        --torch-backend "$VLLM_TORCH_BACKEND" \
        "$VLLM_INSTALL_PACKAGE"
fi
uv pip show --python "$VLLM_VENV/bin/python" vllm | sed -n '1,20p'

export HF_HOME="${HF_HOME:-/work/nvme/bhph/zluo8/hf_home}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export XDG_CACHE_HOME="${SLURM_TMPDIR:-/tmp}/credit_assignment_cache_${SLURM_JOB_ID}"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export CUDA_CACHE_PATH="$XDG_CACHE_HOME/cuda"
mkdir -p "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH"

"$VLLM_VENV/bin/python" scripts/hpc/vllm_token_smoke.py
