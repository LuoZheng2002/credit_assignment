source activate_environment.sh
cargo run --bin bin_browse_session -- \
    --model qwen3.5-4b \
    --dataset-name "deepmath" \
    --num-samples 500