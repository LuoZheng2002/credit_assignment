#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 0 ]]; then
  echo "Usage: $0"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "${SCRIPT_DIR}")"

CONFIG_JSON_PATH="${REPO_ROOT}/train_config/smoke_test_lora_qwen25.json"

bash "${REPO_ROOT}/scripts/train/launch_4gpu.sh" "${CONFIG_JSON_PATH}"
