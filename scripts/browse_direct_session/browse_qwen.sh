

cargo run --bin bin_browse_direct_session -- \
    --model qwen2.5-7b \
    --config-nickname qwen_test \
    --rollout-config-path config/rollout_config.json \
    --temperature-to-accuracy-path config/temperature_to_accuracy_placeholder.json \
    --posterior-hyperparameters-path config/posterior_hyperparameters.json \
    --override-hyperparameters-path config/posterior_hyperparameters_override.json