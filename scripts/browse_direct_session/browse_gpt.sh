

cargo run --bin bin_browse_direct_session -- \
    --model gpt-4o \
    --config-nickname test \
    --rollout-config-path config/rollout_config.json \
    --posterior-hyperparameters-path config/posterior_hyperparameters.json \
    --override-hyperparameters-path config/posterior_hyperparameters_override.json
