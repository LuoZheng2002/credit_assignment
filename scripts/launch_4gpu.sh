#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 0 ]]; then
  echo "Usage: $0"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "${SCRIPT_DIR}")"

CONFIG_JSON_PATH="${REPO_ROOT}/src_py/configs/train_lora_example.json"
MASTER_PORT="29501"

bash "${REPO_ROOT}/src_py/scripts/launch_4gpu_from_config.sh" "${CONFIG_JSON_PATH}" "${MASTER_PORT}"
