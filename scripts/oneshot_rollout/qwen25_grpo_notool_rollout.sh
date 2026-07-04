#!/usr/bin/env bash
set -euo pipefail

# One-shot tree rollout: runs the rollout binary to generate training action logs
# and training trajectories, saved to the shared Modal volume at /volume.
#
# Prerequisites:
#   1. The correct model checkpoint must be present at the expected path on the
#      shared volume (the binary launches its own local inference server).
#   2. Ensure the model checkpoint directory exists before running this script.

uv run -m src_py.modal.launch_modal_oneshot_rollout \
    --model-cli-name qwen25 \
    --max-rollout-concurrency 300 \
    --config-nickname-rollout grpo_notool_rollout \
    --rollout-config-path config/rollout_config_training_grpo_notool.json \
    --dataset-split training \
    --posterior-hyperparameters-path config/posterior_hyperparameters.json \
    --rollout-time-limit-secs 3600 \
    --max-python-processes 2 \
    --advantage-calculation-policy tree-mappo-posterior \
    --positive-advantage-only false \
    --mount-dir "/volume" \
    --num-gpus 1 \
    --gpu-name H200 \
    --modal-time-limit-hrs 1 \
    --ui true
