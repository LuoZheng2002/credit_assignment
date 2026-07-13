RUST_BACKTRACE=1 cargo run --bin bin_direct_tree -- \
    --model-cli-name gpt-4o \
    --max-concurrent-requests 100 \
    --config-nickname test \
    --rollout-config-path config/rollout_config.json \
    --tui-server-port 7878 \
    --first-n-samples 10
