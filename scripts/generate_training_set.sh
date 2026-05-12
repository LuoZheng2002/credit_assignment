source activate_environment.sh

cargo run --bin bin_generate_training_set -- \
    --model qwen2.5-7b \
    --dataset-name deepmath \
    --num-samples 500