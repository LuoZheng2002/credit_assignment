# v3 decisive A/B, arm A: PPO-clip carrier + TreeMAPPO per-segment credit
# (tree rollouts, posterior EM advantages). Same carrier/TOML as the grpo arm;
# only the rollout structure and advantage signal differ.
uv run -m src_py.modal.launch_modal_orchestration \
    --model-cli-name qwen25 \
    --max-rollout-concurrency 300 \
    --config-nickname notool_v3b_tree \
    --validation-rollout-config-path config/rollout_config_validation_notool_t0.json \
    --training-rollout-config-path config/rollout_config_training_notool.json \
    --posterior-hyperparameters-path config/posterior_hyperparameters.json \
    --num-total-epochs 10 \
    --cumulative-avg-abs-advantage-cutoff 0.9 \
    --num-iterations-limit 3 \
    --advantage-calculation-policy tree-mappo-posterior \
    --training-config-common-path config/training/common_lora_v3b.toml \
    --training-time 2400 \
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
