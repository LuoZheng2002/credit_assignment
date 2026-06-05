cargo run --bin bin_browse_direct_session -- \
    --model gemma-3-4b-it \
    --config-nickname std \
    --rollout-config-path config/rollout_config_validation_tool.json \
    --epoch 0 \
    --dataset-split validation \
    --posterior-hyperparameters-path config/posterior_hyperparameters.json \
    --override-hyperparameters-path config/posterior_hyperparameters.json