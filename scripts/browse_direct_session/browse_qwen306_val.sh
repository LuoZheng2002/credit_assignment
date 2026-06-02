cargo run --bin bin_browse_direct_session -- \
    --model qwen3-0.6b \
    --config-nickname tra16_3 \
    --rollout-config-path config/rollout_config_validation.json \
    --posterior-hyperparameters-path config/posterior_hyperparameters.json \
    --epoch 0 \
    --override-hyperparameters-path config/posterior_hyperparameters.json