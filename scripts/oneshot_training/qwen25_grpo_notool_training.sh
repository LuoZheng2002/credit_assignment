#!/usr/bin/env bash
set -euo pipefail

# One-shot training: reads pre-generated training action logs from the shared
# Modal volume, generates training trajectories, then loops through epochs
# (validate → train → write summary).
#
# Prerequisites:
#   1. Run scripts/oneshot_rollout/qwen25_grpo_notool.sh first to populate the
#      training action logs on the shared volume.
#   2. This script launches its own SGLang server internally — no external
#      SGLang endpoint is needed.

uv run -m src_py.modal.launch_modal_oneshot_training \
    --model-cli-name qwen25 \
    --max-rollout-concurrency 300 \
    --config-nickname-rollout grpo_notool_rollout \
    --config-nickname-training grpo_notool_training1 \
    --validation-rollout-config-path config/rollout_config_validation_notool.json \
    --training-rollout-config-path config/rollout_config_training_grpo_notool.json \
    --posterior-hyperparameters-path config/posterior_hyperparameters.json \
    --num-oneshot-epochs 20 \
    --cumulative-avg-abs-advantage-cutoff 0.5 \
    --num-iterations-limit 3 \
    --advantage-calculation-policy tree-mappo-posterior \
    --training-config-common-path config/training/common_lora.toml \
    --oneshot-per-epoch-training-time 600 \
    --validation-rollout-time-limit-secs 1200 \
    --max-python-processes 2 \
    --num-gpus 1 \
    --gpu-name H200 \
    --mount-dir "/volume" \
    --positive-advantage-only false \
    --adam-fp32 false \
    --modal-time-limit-hrs 12 \
    --ui true
