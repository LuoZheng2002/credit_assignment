source activate_environment.sh
RUST_BACKTRACE=1 cargo run --bin bin_direct_tree -- \
    --model-cli-name gpt-4o \
    --max-concurrent-requests 100 \
    --rollout-config-path config/rollout_config.json \
    --temperature-to-accuracy-path config/temperature_to_accuracy_gpt.json \
    --posterior-hyperparameters-path config/posterior_hyperparameters.json \
    --ui false