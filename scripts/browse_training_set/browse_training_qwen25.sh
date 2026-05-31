source .venv/bin/activate
LIBDIR=$(python -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')
export RUSTFLAGS="-C link-arg=-Wl,-rpath,$LIBDIR"

RUST_BACKTRACE=1 cargo run --bin bin_browse_training_set -- \
    --model qwen2.5-7b \
    --config-nickname tra16 \
    --rollout-config-path config/rollout_config_qwen25_temp0_7.json \
    --posterior-hyperparameters-path config/posterior_hyperparameters.json \
    --cumulative-avg-abs-advantage-cutoff 0.5
