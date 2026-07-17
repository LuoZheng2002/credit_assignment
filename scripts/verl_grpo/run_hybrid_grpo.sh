#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: scripts/verl_grpo/run_hybrid_grpo.sh <config-env> [extra verl overrides...]" >&2
    exit 1
fi

CONFIG_ENV="$1"
shift
source "$CONFIG_ENV"

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_ROOT"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen25_hybrid_grpo_lora_r32}"
DATA_DIR="${DATA_DIR:-/work/nvme/bhph/zluo8/credit_assignment/results/verl_grpo_data/$EXPERIMENT_NAME}"
OUTPUT_DIR="${OUTPUT_DIR:-/work/nvme/bhph/zluo8/credit_assignment/results/verl_grpo_runs/$EXPERIMENT_NAME}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}"
NUM_GPUS="${NUM_GPUS:-1}"
VERL_ONE_STEP_OFF="${VERL_ONE_STEP_OFF:-False}"
TRAINING_GPUS="${TRAINING_GPUS:-$NUM_GPUS}"
ROLLOUT_GPUS="${ROLLOUT_GPUS:-1}"
ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE="${ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE:-$NUM_GPUS}"
MODEL_ATTN_IMPLEMENTATION="${MODEL_ATTN_IMPLEMENTATION:-}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-16}"
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}"
ROLLOUT_N="${ROLLOUT_N:-4}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-1024}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
LEARNING_RATE="${LEARNING_RATE:-5e-6}"
LORA_RANK="${LORA_RANK:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:--1}"
TEST_FREQ="${TEST_FREQ:-25}"
SAVE_FREQ="${SAVE_FREQ:-25}"
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-True}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-}"
KL_LOSS_COEF="${KL_LOSS_COEF:-0.001}"
REWARD_OVERRIDE_PREFIX="${VERL_REWARD_OVERRIDE_PREFIX:-custom_reward_function}"
TRAIN_LIMIT="${TRAIN_LIMIT:-}"
VAL_LIMIT="${VAL_LIMIT:-}"
VERL_RUN_DIR="${VERL_RUN_DIR:-$REPO_ROOT}"

mkdir -p "$DATA_DIR" "$OUTPUT_DIR"

PREPARE_ARGS=(
    --output-dir "$DATA_DIR"
)
if [[ -n "$TRAIN_LIMIT" ]]; then
    PREPARE_ARGS+=(--train-limit "$TRAIN_LIMIT")
fi
if [[ -n "$VAL_LIMIT" ]]; then
    PREPARE_ARGS+=(--val-limit "$VAL_LIMIT")
fi
python scripts/verl_grpo/prepare_hybrid_parquet.py "${PREPARE_ARGS[@]}"

python - <<'PY'
import importlib.util
import sys
if importlib.util.find_spec("verl") is None:
    sys.exit("VERL is not installed in the active Python environment. Activate a VERL env or use slurm/verl_grpo_hybrid.slurm to create one.")
PY

COMMON_OVERRIDES=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    algorithm.kl_ctrl.kl_coef=0.0
    data.train_files="$DATA_DIR/train.parquet"
    data.val_files="$DATA_DIR/val.parquet"
    data.prompt_key=prompt
    data.train_batch_size="$TRAIN_BATCH_SIZE"
    data.max_prompt_length="$MAX_PROMPT_LENGTH"
    data.max_response_length="$MAX_RESPONSE_LENGTH"
    data.filter_overlong_prompts=True
    data.truncation=right
    data.shuffle=True
    actor_rollout_ref.model.path="$MODEL_PATH"
    actor_rollout_ref.model.trust_remote_code=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.model.lora_rank="$LORA_RANK"
    actor_rollout_ref.model.lora_alpha="$LORA_ALPHA"
    actor_rollout_ref.model.target_modules=all-linear
    actor_rollout_ref.actor.optim.lr="$LEARNING_RATE"
    actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE"
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$PPO_MICRO_BATCH_SIZE_PER_GPU"
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef="$KL_LOSS_COEF"
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=0.0
    actor_rollout_ref.actor.use_torch_compile=False
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="$LOG_PROB_MICRO_BATCH_SIZE_PER_GPU"
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.n="$ROLLOUT_N"
    actor_rollout_ref.rollout.tensor_model_parallel_size="$ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE"
    actor_rollout_ref.rollout.max_num_batched_tokens="$MAX_NUM_BATCHED_TOKENS"
    actor_rollout_ref.rollout.max_num_seqs="$MAX_NUM_SEQS"
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="$LOG_PROB_MICRO_BATCH_SIZE_PER_GPU"
    reward_model.enable=False
    "$REWARD_OVERRIDE_PREFIX.path=$REPO_ROOT/scripts/verl_grpo/hybrid_reward.py"
    "$REWARD_OVERRIDE_PREFIX.name=compute_score"
    trainer.logger='["console"]'
    trainer.project_name=credit_assignment_verl
    trainer.experiment_name="$EXPERIMENT_NAME"
    trainer.nnodes=1
    trainer.n_gpus_per_node="$NUM_GPUS"
    trainer.total_epochs="$TOTAL_EPOCHS"
    trainer.total_training_steps="$TOTAL_TRAINING_STEPS"
    trainer.val_before_train="$VAL_BEFORE_TRAIN"
    trainer.test_freq="$TEST_FREQ"
    trainer.save_freq="$SAVE_FREQ"
    trainer.critic_warmup=0
    trainer.default_local_dir="$OUTPUT_DIR/checkpoints"
    trainer.resume_mode=disable
)

case "${VERL_ONE_STEP_OFF,,}" in
    1|true|yes)
        COMMON_OVERRIDES+=(
            actor_rollout_ref.hybrid_engine=False
            actor_rollout_ref.actor.fsdp_config.strategy=fsdp2
            actor_rollout_ref.actor.fsdp_config.use_torch_compile=False
            actor_rollout_ref.rollout.free_cache_engine=False
            actor_rollout_ref.rollout.calculate_log_probs=True
            actor_rollout_ref.rollout.checkpoint_engine.backend=nccl
            actor_rollout_ref.rollout.layered_summon=True
            actor_rollout_ref.rollout.load_format=safetensors
            trainer.v1.trainer_mode=separate_async
            trainer.v1.separate_async.parameter_sync_step=1
            trainer.n_gpus_per_node="$TRAINING_GPUS"
            rollout.nnodes=1
            rollout.n_gpus_per_node="$ROLLOUT_GPUS"
        )
        VERL_MAIN_MODULE="${VERL_MAIN_MODULE:-verl.experimental.one_step_off_policy.main_ppo}"
        ;;
    *)
        VERL_MAIN_MODULE="${VERL_MAIN_MODULE:-verl.trainer.main_ppo}"
        ;;
esac


if [[ -n "$GPU_MEMORY_UTILIZATION" ]]; then
    COMMON_OVERRIDES+=(actor_rollout_ref.rollout.gpu_memory_utilization="$GPU_MEMORY_UTILIZATION")
fi
if [[ -n "$MODEL_ATTN_IMPLEMENTATION" ]]; then
    COMMON_OVERRIDES+=(+actor_rollout_ref.model.override_config.attn_implementation="$MODEL_ATTN_IMPLEMENTATION")
fi

set -x
(
    cd "$VERL_RUN_DIR"
    python -m "$VERL_MAIN_MODULE" "${COMMON_OVERRIDES[@]}" "$@"
)
