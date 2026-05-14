source activate_environment.sh
RUST_BACKTRACE=1 cargo run --bin bin_tree -- \
    --dataset-name deepmath \
    --num-samples 10 \
    --model gpt-4o \
    --vllm-ports 8000 \
    --ui false