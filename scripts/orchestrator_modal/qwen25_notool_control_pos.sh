# Positive-control run, take 2: positive-advantage-only (bounded, weighted-SFT
# objective) after the +/- advantage objective diverged at LR>=4e-5 within the
# first on-policy pass (CE 0.05 -> 76, grad_norm > 100). Same pinned questions.
uv run -m src_py.modal.launch_modal_orchestration \
    --model-cli-name qwen25 \
    --max-rollout-concurrency 300 \
    --config-nickname notool_control_pos \
    --validation-rollout-config-path config/rollout_config_validation_notool_t0.json \
    --training-rollout-config-path config/rollout_config_training_notool_control.json \
    --posterior-hyperparameters-path config/posterior_hyperparameters.json \
    --num-total-epochs 4 \
    --cumulative-avg-abs-advantage-cutoff 1.0 \
    --num-iterations-limit 4 \
    --advantage-calculation-policy tree-mappo-posterior \
    --training-config-common-path config/training/common_lora_control.toml \
    --training-time 2400 \
    --training-rollout-time-limit-secs 600 \
    --validation-rollout-time-limit-secs 600 \
    --max-python-processes 4 \
    --num-gpus 1 \
    --gpu-name H200 \
    --mount-dir "/volume" \
    --sft false \
    --positive-advantage-only true \
    --keep-action-logs true \
    --adam-fp32 false \
    --ui true
