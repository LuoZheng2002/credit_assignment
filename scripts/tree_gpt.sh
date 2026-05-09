

cargo run --bin bin_tree -- \
    --dataset-name deepmath \
    --num-samples 2 \
    --model gpt-4o \
    --vllm-port 8000 \
    --max-tasks 1000 \
    --take-over-mode-decision true