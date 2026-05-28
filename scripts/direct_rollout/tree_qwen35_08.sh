source .venv/bin/activate
LIBDIR=$(python -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')
export RUSTFLAGS="-C link-arg=-Wl,-rpath,$LIBDIR"

RUST_BACKTRACE=1 cargo run --bin bin_direct_tree -- \
    --model-cli-name qwen3.5-0.8b \
    --qwen-sglang-port 30000 \
    --max-concurrent-questions 200 \
    --config-nickname tra16 \
    --rollout-config-path config/rollout_config_qwen35_08_temp0_7.json \
    --posterior-hyperparameters-path config/posterior_hyperparameters.json \
    --epoch 0 \
    --max-sqlite-connections 50 \
    --ui false \
    --first-n-samples 200

