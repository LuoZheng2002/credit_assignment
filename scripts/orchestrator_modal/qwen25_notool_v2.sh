# v2 main run: ~3x rollout data per epoch, cutoff 0.9 (keeps most mixed-tree
# trajectories), single on-policy pass, LoRA LR 2e-5 with MLP modules,
# greedy (temp=0) validation for low-noise epoch-to-epoch comparison.
uv run -m src_py.modal.launch_modal_orchestration \
    --model-cli-name qwen25 \
    --max-rollout-concurrency 300 \
    --config-nickname notool_v2 \
    --validation-rollout-config-path config/rollout_config_validation_notool_t0.json \
    --training-rollout-config-path config/rollout_config_training_notool.json \
    --posterior-hyperparameters-path config/posterior_hyperparameters.json \
    --num-total-epochs 10 \
    --cumulative-avg-abs-advantage-cutoff 0.9 \
    --num-iterations-limit 1 \
    --advantage-calculation-policy tree-mappo-posterior \
    --training-config-common-path config/training/common_lora_v2.toml \
    --training-time 1800 \
    --training-rollout-time-limit-secs 1800 \
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
