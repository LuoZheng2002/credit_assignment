# source .venv/bin/activate
LIBDIR=$(python -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')
export RUSTFLAGS="-C link-arg=-Wl,-rpath,$LIBDIR"

RUST_BACKTRACE=1 cargo run --bin bin_browse_training_set -- \
    --model qwen2.5-7b \
    --config-nickname tra16_4 \
    --rollout-config-path config/rollout_config_training_std.json \
    --posterior-hyperparameters-path config/posterior_hyperparameters.json \
    --epoch 0 \
    --advantage-calculation-policy tree-mappo-posterior \
    --cumulative-avg-abs-advantage-cutoff 0.5
