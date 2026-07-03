# v2 positive-only ablation: same as notool_v2 but negative advantages are
# clamped to zero (weighted-SFT-like objective; inherently stable at higher LR).
uv run -m src_py.modal.launch_modal_orchestration \
    --model-cli-name qwen25 \
    --max-rollout-concurrency 300 \
    --config-nickname notool_v2_positive \
    --validation-rollout-config-path config/rollout_config_validation_notool_t0.json \
    --training-rollout-config-path config/rollout_config_training_notool.json \
    --posterior-hyperparameters-path config/posterior_hyperparameters.json \
    --num-total-epochs 10 \
    --cumulative-avg-abs-advantage-cutoff 0.9 \
    --num-iterations-limit 1 \
    --advantage-calculation-policy tree-mappo-posterior \
    --training-config-common-path config/training/common_lora_v2_positive.toml \
    --training-time 1800 \
    --training-rollout-time-limit-secs 1800 \
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
