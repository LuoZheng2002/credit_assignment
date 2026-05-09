source activate_environment.sh
cargo run --bin bin_rollout_pipeline -- \
    --dataset-name "deepmath" \
 --num-samples 10 \
 --model qwen2.5-7b \
 --vllm-port 8000\
 --take-over-mode-decision true