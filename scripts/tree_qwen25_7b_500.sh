source activate_environment.sh
cargo run --bin bin_tree -- \
    --dataset-name "deepmath" \
 --num-samples 500 \
 --model qwen2.5-7b \
 --vllm-ports 8000 \
 --ui true