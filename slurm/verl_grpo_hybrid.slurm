#!/bin/bash

#SBATCH --job-name=verl_grpo_hybrid
#SBATCH --output=slurm/logs/verl_grpo_%j.out
#SBATCH --error=slurm/logs/verl_grpo_%j.err
#SBATCH --cpus-per-task=32
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --account=bfsl-delta-gpu
#SBATCH --time=04:00:00

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: sbatch --gres=gpu:nvidia_a100:<num-gpus> slurm/verl_grpo_hybrid.slurm <config-env> [extra verl overrides...]" >&2
    exit 1
fi

CONFIG_ENV="$1"
shift
NOTIFY_START_MSG="${NOTIFY_START_MSG:-}"
NOTIFY_END_MSG="${NOTIFY_END_MSG:-VERL GRPO hybrid job finished}"

REPO_ROOT="$SLURM_SUBMIT_DIR"
cd "$REPO_ROOT"

if [[ -n "${VERL_LOAD_MODULE:-}" ]]; then
    module load "$VERL_LOAD_MODULE"
fi

if [[ -n "$NOTIFY_END_MSG" ]]; then
    trap 'python research-utility/scripts/pushover_notify.py "$NOTIFY_END_MSG"' EXIT
fi
if [[ -n "$NOTIFY_START_MSG" ]]; then
    python research-utility/scripts/pushover_notify.py "$NOTIFY_START_MSG"
fi

VERL_VENV="${VERL_VENV:-/work/nvme/bhph/zluo8/credit_assignment/venvs/verl-grpo}"
if [[ ! -x "$VERL_VENV/bin/python" ]]; then
    python -m venv "$VERL_VENV"
    "$VERL_VENV/bin/python" -m pip install --upgrade pip wheel 'setuptools<82'
    "$VERL_VENV/bin/python" -m pip install 'verl[vllm]' datasets pyarrow
    FLASH_ATTN_WHEEL="${FLASH_ATTN_WHEEL:-https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp39-cp39-linux_x86_64.whl}"
    "$VERL_VENV/bin/python" -m pip install "$FLASH_ATTN_WHEEL"
    "$VERL_VENV/bin/python" -m pip install --upgrade 'setuptools<82'
fi
source "$VERL_VENV/bin/activate"

export TOKENIZERS_PARALLELISM=false
export RAY_DEDUP_LOGS=0
export HYDRA_FULL_ERROR=1

bash scripts/verl_grpo/run_hybrid_grpo.sh "$CONFIG_ENV" "$@"
