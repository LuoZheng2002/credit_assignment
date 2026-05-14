source activate_environment.sh
cargo run --bin bin_tree -- \
    --dataset-name "deepmath" \
 --num-samples 500 \
 --model qwen3.5-4b \
 --vllm-ports 8002 \
 --ui true