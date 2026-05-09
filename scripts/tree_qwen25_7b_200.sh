source activate_environment.sh
cargo run --bin bin_tree -- \
    --dataset-name "deepmath" \
 --num-samples 200 \
 --model qwen2.5-7b \
 --vllm-port 8000