# Positive-control run: every epoch rolls out the SAME question segment
# (question_rotation_seed=0), trains with SFT-scale LR (1e-4) for 4 passes
# over all mixed-tree trajectories (cutoff 1.0). Success criterion: epoch 1-3
# training_rollout_accuracies rise clearly above epoch 0 on these questions.
uv run -m src_py.modal.launch_modal_orchestration \
    --model-cli-name qwen25 \
    --max-rollout-concurrency 300 \
    --config-nickname notool_control \
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
    --positive-advantage-only false \
    --keep-action-logs true \
    --adam-fp32 false \
    --ui true
