source activate_environment.sh
RUST_BACKTRACE=1 cargo run --bin bin_direct_tree -- \
    --model-cli-name qwen2.5-7b \
    --qwen-api-backend vllm \
    --qwen-vllm-port 8000 \
    --max-concurrent-requests 100 \
    --config-nickname qwen_test \
    --rollout-config-path config/rollout_config.json \
    --temperature-to-accuracy-path config/temperature_to_accuracy_placeholder.json \
    --posterior-hyperparameters-path config/posterior_hyperparameters.json \
    --ui false \
    --first-n-samples 10