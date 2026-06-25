cargo run --bin bin_browse_direct_session -- \
    --model qwen25 \
    --config-nickname tra16 \
    --rollout-config-path config/rollout_config_qwen25_temp0_7.json \
    --posterior-hyperparameters-path config/posterior_hyperparameters.json \
    --override-hyperparameters-path config/posterior_hyperparameters.json
