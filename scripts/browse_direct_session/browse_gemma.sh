cargo run --bin bin_browse_direct_session -- \
    --model gemma \
    --config-nickname notool \
    --rollout-config-path config/rollout_config_validation_notool.json \
    --epoch 0 \
    --dataset-split validation \
    --posterior-hyperparameters-path config/posterior_hyperparameters.json \
    --override-hyperparameters-path config/posterior_hyperparameters.json
