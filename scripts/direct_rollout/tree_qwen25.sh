source activate_environment.sh
source .venv/bin/activate
LIBDIR=$(python -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')

RUSTFLAGS="-C link-arg=-Wl,-rpath,$LIBDIR" RUST_BACKTRACE=1 cargo run --bin bin_direct_tree -- \
    --model-cli-name qwen2.5-7b \
    --qwen-api-backend sglang \
    --qwen-sglang-port 30000 \
    --max-concurrent-requests 100 \
    --config-nickname qwen_test \
    --rollout-config-path config/rollout_config.json \
    --temperature-to-accuracy-path config/temperature_to_accuracy_placeholder.json \
    --posterior-hyperparameters-path config/posterior_hyperparameters.json \
    --ui false \
    --first-n-samples 10