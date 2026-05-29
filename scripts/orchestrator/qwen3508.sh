set -euo pipefail

source pyprojects/common/.venv/bin/activate
LIBDIR=$(python -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')
export RUSTFLAGS="-C link-arg=-Wl,-rpath,$LIBDIR"

RUST_BACKTRACE=1 cargo run --bin bin_orchestrator -- \
    --model-cli-name qwen3.5-0.8b \
    --max-rollout-concurrency 200 \
    --config-nickname tra16 \
    --validation-rollout-config-path config/rollout_config_validation.json \
    --training-rollout-config-path config/rollout_config_training_qwen35_08_temp0_7.json \
    --posterior-hyperparameters-path config/posterior_hyperparameters.json \
    --num-total-epochs 3 \
    --max-num-training-trajectories 100 \
    --advantage-calculation-policy tree-mappo-posterior \
    --training-config-common-path config/training/common.toml \
    --first-n-training-samples 100 \
    --first-n-rollout-samples 10 \
    --max-sqlite-connections 1 \
    --sglang-server-log-path logs/sglang_server.txt \
    --message-log-path logs/messages.txt \
    --num-iterations 3 \
    --num-gpus 1 \
    --ui true