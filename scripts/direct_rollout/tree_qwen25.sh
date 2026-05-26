
source .venv/bin/activate
LIBDIR=$(python -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')
export RUSTFLAGS="-C link-arg=-Wl,-rpath,$LIBDIR"

RUST_BACKTRACE=1 cargo run --bin bin_direct_tree -- \
    --model-cli-name qwen2.5-7b \
    --qwen-api-backend sglang \
    --qwen-sglang-port 30000 \
    --max-concurrent-requests 200 \
    --config-nickname tra16 \
    --rollout-config-path config/rollout_config_qwen25_temp0_7.json \
    --posterior-hyperparameters-path config/posterior_hyperparameters.json \
    --ui false \
    --first-n-samples 15000

