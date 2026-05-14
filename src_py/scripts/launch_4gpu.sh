#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 18 ]]; then
  echo "Usage: $0 <training_plan> <model_name_or_path> <tokenized_sqlite_path> <batch_sqlite_path> <output_dir> <advantage_clip> <learning_rate> <weight_decay> <num_epochs> <grad_accum_steps> <log_interval_steps> <save_interval_steps> <lora_rank> <lora_alpha> <lora_dropout> <lora_target_modules_csv> <seed> <master_port> [resume_checkpoint_tag]"
  exit 1
fi

TRAINING_PLAN="$1"
MODEL_NAME_OR_PATH="$2"
TOKENIZED_SQLITE_PATH="$3"
BATCH_SQLITE_PATH="$4"
OUTPUT_DIR="$5"
ADVANTAGE_CLIP="$6"
LEARNING_RATE="$7"
WEIGHT_DECAY="$8"
NUM_EPOCHS="${9}"
GRAD_ACCUM_STEPS="${10}"
LOG_INTERVAL_STEPS="${11}"
SAVE_INTERVAL_STEPS="${12}"
LORA_RANK="${13}"
LORA_ALPHA="${14}"
LORA_DROPOUT="${15}"
LORA_TARGET_MODULES_CSV="${16}"

if [[ $# -eq 18 ]]; then
  SEED="${17}"
  MASTER_PORT="${18}"
  RESUME_CHECKPOINT_TAG="auto"
else
  SEED="${17}"
  MASTER_PORT="${18}"
  RESUME_CHECKPOINT_TAG="${19}"
fi

torchrun --nproc_per_node 4 --master_port "${MASTER_PORT}" src_py/train/main.py \
  --training-plan "${TRAINING_PLAN}" \
  --model-name-or-path "${MODEL_NAME_OR_PATH}" \
  --tokenized-sqlite-path "${TOKENIZED_SQLITE_PATH}" \
  --batch-sqlite-path "${BATCH_SQLITE_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
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
