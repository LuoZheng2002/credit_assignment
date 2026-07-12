#!/usr/bin/env bash
set -euo pipefail

# One-shot training: reads pre-generated training trajectories from the shared
# Modal volume, then loops through epochs (validate → train → write summary).
#
# Prerequisites:
#   1. Run scripts/oneshot_rollout/qwen25_grpo_notool.sh first to populate the
#      training action logs on the shared volume.
#   2. This script launches its own SGLang server internally — no external
#      SGLang endpoint is needed.

uv run -m src_py.modal.launch_modal_oneshot_training \
    --model-cli-name qwen25 \
    --max-rollout-concurrency 300 \
    --config-nickname-training grpo_notool_training_full \
    --config-nickname-rollout grpo_notool_rollout \
    --num-oneshot-epochs 20 \
    --num-iterations-limit 3 \
    --training-config-common-path config/training/common_ddp.toml \
    --oneshot-per-epoch-training-time 600 \
    --validation-rollout-time-limit-secs 1200 \
    --max-python-processes 2 \
    --num-gpus 1 \
    --gpu-name H200 \
    --mount-dir "/volume" \
    --rollout-mount-dir "/rollout_volume" \
    --modal-time-limit-hrs 4 \
    --ui true
