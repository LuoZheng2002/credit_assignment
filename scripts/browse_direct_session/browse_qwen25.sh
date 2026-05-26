source .venv/bin/activate
LIBDIR=$(python -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')
export RUSTFLAGS="-C link-arg=-Wl,-rpath,$LIBDIR"

cargo run --bin bin_browse_direct_session -- \
    --model qwen2.5-7b \
    --config-nickname tra16 \
    --rollout-config-path config/rollout_config_qwen25_temp0_7.json \
    --posterior-hyperparameters-path config/posterior_hyperparameters.json \
    --override-hyperparameters-path config/posterior_hyperparameters.json