source activate_environment.sh

cargo run --bin bin_generate_training_set -- \
    --model gpt-4o \
    --dataset-name deepmath \
    --num-samples 2