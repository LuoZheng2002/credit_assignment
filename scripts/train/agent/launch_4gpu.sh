#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <config_json_path>"
  exit 1
fi

CONFIG_JSON_PATH="$1"
MASTER_PORT="${MASTER_PORT:-29501}"

torchrun --nproc_per_node 4 --master_port "${MASTER_PORT}" src_py/train/main_from_config.py \
  --config-json-path "${CONFIG_JSON_PATH}"
