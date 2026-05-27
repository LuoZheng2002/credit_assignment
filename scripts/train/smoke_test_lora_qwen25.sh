#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 1 ]]; then
  echo "Usage: $0 [num_gpus]"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$(dirname "${SCRIPT_DIR}")")"
NUM_GPUS="${1:-${NUM_GPUS:-1}}"
MASTER_PORT="${MASTER_PORT:-29501}"

CONFIG_JSON_PATH="${REPO_ROOT}/train_config/smoke_test_lora_qwen25.json"

torchrun --nproc_per_node "${NUM_GPUS}" --master_port "${MASTER_PORT}" src_py/train/main_from_config.py \
  --config-json-path "${CONFIG_JSON_PATH}"
