cargo run --bin bin_browse_direct_session -- \
    --model qwen3.5-0.8b \
    --config-nickname tra16 \
    --rollout-config-path config/rollout_config_qwen35_08_temp0_7.json \
    --posterior-hyperparameters-path config/posterior_hyperparameters.json \
    --epoch 0 \
    --override-hyperparameters-path config/posterior_hyperparameters.json