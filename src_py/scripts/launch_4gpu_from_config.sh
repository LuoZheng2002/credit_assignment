#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <config_json_path> <master_port>"
  exit 1
fi

CONFIG_JSON_PATH="$1"
MASTER_PORT="$2"

torchrun --nproc_per_node 4 --master_port "${MASTER_PORT}" src_py/train/main_from_config.py \
  --config-json-path "${CONFIG_JSON_PATH}"
