source activate_environment.sh
RUST_BACKTRACE=1 cargo run --bin bin_tree -- \
    --dataset-name deepmath \
    --num-samples 2 \
    --model gpt-4o \
    --vllm-port 8000 