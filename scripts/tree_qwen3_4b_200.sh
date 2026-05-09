source activate_environment.sh
cargo run --bin bin_rollout_pipeline -- \
    --dataset-name "deepmath" \
 --num-samples 200 \
 --model qwen3-4b \
 --vllm-port 8001