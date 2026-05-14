#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 15 ]]; then
  echo "Usage: $0 <model_name_or_path> <tokenized_sqlite_path> <batch_sqlite_path> <deepspeed_config_path> <output_dir> <pad_token_id> <advantage_clip> <learning_rate> <weight_decay> <num_epochs> <grad_accum_steps> <log_interval_steps> <save_interval_steps> <seed> <master_port>"
  exit 1
fi

MODEL_NAME_OR_PATH="$1"
TOKENIZED_SQLITE_PATH="$2"
BATCH_SQLITE_PATH="$3"
DEEPSPEED_CONFIG_PATH="$4"
OUTPUT_DIR="$5"
PAD_TOKEN_ID="$6"
ADVANTAGE_CLIP="$7"
LEARNING_RATE="$8"
WEIGHT_DECAY="$9"
NUM_EPOCHS="${10}"
GRAD_ACCUM_STEPS="${11}"
LOG_INTERVAL_STEPS="${12}"
SAVE_INTERVAL_STEPS="${13}"
SEED="${14}"
MASTER_PORT="${15}"

deepspeed --num_gpus 4 --master_port "${MASTER_PORT}" src_py/train/main.py \
  --model-name-or-path "${MODEL_NAME_OR_PATH}" \
  --tokenized-sqlite-path "${TOKENIZED_SQLITE_PATH}" \
  --batch-sqlite-path "${BATCH_SQLITE_PATH}" \
  --deepspeed-config-path "${DEEPSPEED_CONFIG_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --pad-token-id "${PAD_TOKEN_ID}" \
  --advantage-clip "${ADVANTAGE_CLIP}" \
  --learning-rate "${LEARNING_RATE}" \
  --weight-decay "${WEIGHT_DECAY}" \
  --num-epochs "${NUM_EPOCHS}" \
  --grad-accum-steps "${GRAD_ACCUM_STEPS}" \
  --log-interval-steps "${LOG_INTERVAL_STEPS}" \
  --save-interval-steps "${SAVE_INTERVAL_STEPS}" \
  --seed "${SEED}"
