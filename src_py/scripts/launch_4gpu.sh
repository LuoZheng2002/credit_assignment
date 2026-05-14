#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 19 ]]; then
  echo "Usage: $0 <training_plan> <model_name_or_path> <tokenized_sqlite_path> <batch_sqlite_path> <output_dir> <pad_token_id> <advantage_clip> <learning_rate> <weight_decay> <num_epochs> <grad_accum_steps> <log_interval_steps> <save_interval_steps> <lora_rank> <lora_alpha> <lora_dropout> <lora_target_modules_csv> <seed> <master_port> [resume_checkpoint_tag]"
  exit 1
fi

TRAINING_PLAN="$1"
MODEL_NAME_OR_PATH="$2"
TOKENIZED_SQLITE_PATH="$3"
BATCH_SQLITE_PATH="$4"
OUTPUT_DIR="$5"
PAD_TOKEN_ID="$6"
ADVANTAGE_CLIP="$7"
LEARNING_RATE="$8"
WEIGHT_DECAY="$9"
NUM_EPOCHS="${10}"
GRAD_ACCUM_STEPS="${11}"
LOG_INTERVAL_STEPS="${12}"
SAVE_INTERVAL_STEPS="${13}"
LORA_RANK="${14}"
LORA_ALPHA="${15}"
LORA_DROPOUT="${16}"
LORA_TARGET_MODULES_CSV="${17}"

if [[ $# -eq 19 ]]; then
  SEED="${18}"
  MASTER_PORT="${19}"
  RESUME_CHECKPOINT_TAG="auto"
else
  SEED="${18}"
  MASTER_PORT="${19}"
  RESUME_CHECKPOINT_TAG="${20}"
fi

torchrun --nproc_per_node 4 --master_port "${MASTER_PORT}" src_py/train/main.py \
  --training-plan "${TRAINING_PLAN}" \
  --model-name-or-path "${MODEL_NAME_OR_PATH}" \
  --tokenized-sqlite-path "${TOKENIZED_SQLITE_PATH}" \
  --batch-sqlite-path "${BATCH_SQLITE_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --pad-token-id "${PAD_TOKEN_ID}" \
  --advantage-clip "${ADVANTAGE_CLIP}" \
  --learning-rate "${LEARNING_RATE}" \
  --weight-decay "${WEIGHT_DECAY}" \
  --num-epochs "${NUM_EPOCHS}" \
  --grad-accum-steps "${GRAD_ACCUM_STEPS}" \
  --log-interval-steps "${LOG_INTERVAL_STEPS}" \
  --save-interval-steps "${SAVE_INTERVAL_STEPS}" \
  --lora-rank "${LORA_RANK}" \
  --lora-alpha "${LORA_ALPHA}" \
  --lora-dropout "${LORA_DROPOUT}" \
  --lora-target-modules-csv "${LORA_TARGET_MODULES_CSV}" \
  --resume-checkpoint-tag "${RESUME_CHECKPOINT_TAG}" \
  --seed "${SEED}"
