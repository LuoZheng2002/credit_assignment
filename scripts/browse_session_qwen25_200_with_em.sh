source activate_environment.sh
cargo run --bin bin_browse_session -- \
    --file results/qwen2.5-7b/rollout/deepmath_trajectory_200.jsonl \
    --em-fit-file results/qwen2.5-7b/rollout/deepmath_em_fit_per_tree_200.jsonl
