RUST_BACKTRACE=1 cargo run --bin bin_get_accuracy -- \
    --model-cli-name qwen2.5-7b \
    --config-nickname accuracy0_7 \
    --rollout-config-path config/rollout_config_accuracy0_7.json \
    --posterior-hyperparameters-path config/posterior_hyperparameters.json