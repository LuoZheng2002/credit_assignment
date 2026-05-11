source activate_environment.sh
cargo run --bin bin_em_fit_trees -- \
    --trees-file results/qwen2.5-7b/rollout/deepmath_trajectory_200.jsonl \
    --output-file results/qwen2.5-7b/rollout/deepmath_em_fit_per_tree_200.jsonl \
    --em-fit-meta-file results/qwen2.5-7b/rollout/deepmath_em_fit_meta_200.json
