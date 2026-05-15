source activate_environment.sh
cargo run --bin bin_em_fit_trees -- \
    --model qwen3.5-4b \
    --dataset-name deepmath \
    --num-samples 200
