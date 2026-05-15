source activate_environment.sh
cargo run --bin bin_browse_session -- \
    --file results/qwen3.5-4b/rollout/deepmath_trajectory_200.jsonl \
    --correctness-file results/qwen3.5-4b/rollout/deepmath_correctness_200.jsonl