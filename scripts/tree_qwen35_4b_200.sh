source activate_environment.sh
cargo run --bin bin_tree -- \
    --dataset-name "deepmath" \
 --num-samples 200 \
 --model qwen3.5-4b \
 --vllm-port 8002