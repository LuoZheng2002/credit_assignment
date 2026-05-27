#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <num_gpus> <config_toml_path>"
  exit 1
fi

NUM_GPUS="$1"
CONFIG_TOML_PATH="$2"
MASTER_PORT="${MASTER_PORT:-29501}"

torchrun --nproc_per_node "${NUM_GPUS}" --master_port "${MASTER_PORT}" src_py/train/main_from_config.py \
  --config-toml-path "${CONFIG_TOML_PATH}"
