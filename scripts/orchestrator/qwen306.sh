set -euo pipefail

source pyprojects/common/.venv/bin/activate
LIBDIR=$(python -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')
export RUSTFLAGS="-C link-arg=-Wl,-rpath,$LIBDIR"

RUST_BACKTRACE=1 cargo run --bin bin_orchestrator -- \
    --model-cli-name qwen3-0.6b \
    --max-rollout-concurrency 200 \
    --config-nickname tra16 \
    --validation-rollout-config-path config/rollout_config_validation.json \
    --training-rollout-config-path config/rollout_config_training_qwen3_06_temp0_7.json \
    --posterior-hyperparameters-path config/posterior_hyperparameters.json \
    --num-total-epochs 3 \
    --cumulative-avg-abs-advantage-cutoff 0.5 \
    --num-iterations-limit 5 \
    --advantage-calculation-policy tree-mappo-posterior \
    --training-config-common-path config/training/common.toml \
    --training-time 300 \
    --training-rollout-time-limit-secs 300 \
    --validation-rollout-time-limit-secs 300 \
    --max-sqlite-connections 1 \
    --sglang-server-log-path logs/sglang_server.txt \
    --message-log-path logs/messages.txt \
    --num-gpus 1 \
    --ui true
