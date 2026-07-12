cargo run --bin bin_browse_direct_session -- \
    --model qwen25 \
    --config-nickname notool \
    --rollout-config-path config/rollout_config_testing_notool.json \
    --override-hyperparameters-path config/posterior_hyperparameters.json \
    --dataset-split testing \
    --epoch 0
