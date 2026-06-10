#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <num_gpus> <job_folder_path>"
  exit 1
fi

NUM_GPUS="$1"
JOB_FOLDER_PATH="$2"
MASTER_PORT="${MASTER_PORT:-29501}"

uv run torchrun --nproc_per_node "${NUM_GPUS}" --master_port "${MASTER_PORT}" -m src_py.train.main \
  --job-folder-path "${JOB_FOLDER_PATH}"
