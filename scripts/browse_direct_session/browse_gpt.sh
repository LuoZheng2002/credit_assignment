

cargo run --bin bin_browse_direct_session -- \
    --model gpt-4o \
    --config-nickname test \
    --rollout-config-path config/rollout_config.json \
    --temperature-to-accuracy-path config/temperature_to_accuracy_placeholder.json \
    --posterior-hyperparameters-path config/posterior_hyperparameters.json