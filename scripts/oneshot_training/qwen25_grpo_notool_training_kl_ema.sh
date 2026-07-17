#!/usr/bin/env bash
set -euo pipefail

# One-shot training: reads pre-generated training trajectories from the shared
# Modal volume, then loops through epochs (validate → train → write summary).
#
# Prerequisites:
#   1. Run the oneshot rollout step, then the oneshot generation step, to
#      populate the shared generation volume.
#   2. This script launches its own SGLang server internally — no external
#      SGLang endpoint is needed.

uv run -m src_py.modal.launch_modal_oneshot_training \
    --model-cli-name qwen25 \
    --config-nickname-training grpo_notool_training_kl_ema \
    --config-nickname-generation grpo_notool_generation \
    --num-oneshot-epochs 20 \
    --num-iterations-limit 3 \
    --training-config-common-path config/training/common_fsdp_kl_ema.toml \
    --oneshot-per-epoch-training-time 600 \
    --validation-rollout-time-limit-secs 1200 \
    --num-gpus 2 \
    --gpu-name H200 \
    --mount-dir "/volume" \
    --generation-mount-dir "/generation_volume" \
    --modal-time-limit-hrs 6 \
