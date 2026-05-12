source activate_environment.sh
cargo run --bin bin_em_fit_trees -- \
    --model qwen2.5-7b \
    --dataset-name deepmath \
    --num-samples 200
