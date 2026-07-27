#!/usr/bin/env bash
set -euo pipefail

# One-shot generation: reads rollout action logs and materializes training
# trajectories into the shared generation volume.
#
# Prerequisite:
#   1. Run the oneshot rollout step first so the rollout volume contains action logs.

uv run --project pyprojects/minimal -m src_py.modal.launch_modal_oneshot_generation \
    --model-cli-name qwen25 \
    --config-nickname-rollout grpo_notool_rollout \
    --config-nickname-generation grpo_notool_generation \
    --rollout-config-path config/rollout_config_training_grpo.json \
    --use-tool false \
    --epoch 0 \
    --training-advantage-policy TreeMappoPosterior \
    --positive-advantage-only false \
    --rollout-mount-dir "/rollout_volume" \
    --generation-mount-dir "/generation_volume" \
    --num-gpus 1 \
    --gpu-name H200 \
    --modal-time-limit-hrs 2
