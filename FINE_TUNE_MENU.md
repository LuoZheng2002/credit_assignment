# Fine-Tune Menu

This menu shows the practical commands to train and resume with the current Python training stack.

## 0) Prerequisites

- Python env exists at `.venv`.
- Dependencies are installed from `requirements.txt`.
- Tokenized sqlite and batch sqlite are prepared.
- You are at repo root: `credit_assignment/`.

Example setup:

```bash
.venv/bin/pip install -r requirements.txt
```

## 1) Choose Training Plan

- `lora_current`: current default plan (LoRA adapters).
- `full_fsdp_backup`: backup plan (full-model FSDP).

## 2) Start Training (4 GPUs, existing launcher)

Use `src_py/scripts/launch_4gpu.sh`.

Required positional args:

1. `training_plan`
2. `model_name_or_path`
3. `tokenized_sqlite_path`
4. `batch_sqlite_path`
5. `output_dir`
6. `advantage_clip`
7. `learning_rate`
8. `weight_decay`
9. `num_epochs`
10. `grad_accum_steps`
11. `log_interval_steps`
12. `save_interval_steps`
13. `lora_rank`
14. `lora_alpha`
15. `lora_dropout`
16. `lora_target_modules_csv`
17. `seed`
18. `master_port`
19. `[resume_checkpoint_tag]` (optional; default: `auto`)

Example (LoRA):

```bash
bash src_py/scripts/launch_4gpu.sh \
  lora_current \
  Qwen/Qwen2.5-7B-Instruct \
  /path/to/tokenized.sqlite \
  /path/to/batches.sqlite \
  /path/to/output_run \
  3.0 \
  1e-5 \
  0.01 \
  3 \
  4 \
  10 \
  100 \
  64 \
  128 \
  0.05 \
  q_proj,k_proj,v_proj,o_proj \
  42 \
  29501
```

Example (FSDP backup):

```bash
bash src_py/scripts/launch_4gpu.sh \
  full_fsdp_backup \
  Qwen/Qwen2.5-7B-Instruct \
  /path/to/tokenized.sqlite \
  /path/to/batches.sqlite \
  /path/to/output_run_fsdp \
  3.0 \
  1e-5 \
  0.01 \
  3 \
  4 \
  10 \
  100 \
  64 \
  128 \
  0.05 \
  q_proj,k_proj,v_proj,o_proj \
  42 \
  29502
```

## 3) Resume Behavior

Checkpoint pointer file:

- `output_dir/latest_checkpoint.txt`

Each checkpoint directory (for example `global_step_100/`) includes:

- `model_state.pt`
- `optimizer_state.rank{rank}.pt`
- `training_state.rank{rank}.pt`

Resume modes via `resume_checkpoint_tag`:

- `auto` (default): resume from latest if pointer exists, else start fresh.
- `latest`: require pointer file and resume from it.
- `none`: always start fresh.
- explicit tag: e.g. `global_step_100`.

Resume examples:

```bash
# Default auto-resume (omit last arg)
bash src_py/scripts/launch_4gpu.sh ... 42 29501

# Force latest (must exist)
bash src_py/scripts/launch_4gpu.sh ... 42 29501 latest

# Force specific checkpoint tag
bash src_py/scripts/launch_4gpu.sh ... 42 29501 global_step_100

# Force fresh run
bash src_py/scripts/launch_4gpu.sh ... 42 29501 none
```

## 4) Direct Python Entry (without launcher)

You can also call `src_py/train/main.py` directly.

```bash
torchrun --nproc_per_node 4 --master_port 29501 src_py/train/main.py \
  --training-plan lora_current \
  --model-name-or-path Qwen/Qwen2.5-7B-Instruct \
  --tokenized-sqlite-path /path/to/tokenized.sqlite \
  --batch-sqlite-path /path/to/batches.sqlite \
  --output-dir /path/to/output_run \
  --advantage-clip 3.0 \
  --learning-rate 1e-5 \
  --weight-decay 0.01 \
  --num-epochs 3 \
  --grad-accum-steps 4 \
  --log-interval-steps 10 \
  --save-interval-steps 100 \
  --lora-rank 64 \
  --lora-alpha 128 \
  --lora-dropout 0.05 \
  --lora-target-modules-csv q_proj,k_proj,v_proj,o_proj \
  --seed 42 \
  --resume-checkpoint-tag auto
```

## 5) JSON Config Entry (recommended for many hyperparameters)

You can move all training fields into a JSON file and launch with a minimal command.

Example config file:

- `src_py/configs/train_lora_example.json`

Launch with JSON:

```bash
bash src_py/scripts/launch_4gpu_from_config.sh src_py/configs/train_lora_example.json 29501
```

Direct `torchrun` with JSON:

```bash
torchrun --nproc_per_node 4 --master_port 29501 src_py/train/main_from_config.py \
  --config-json-path src_py/configs/train_lora_example.json
```

JSON schema rule:

- The JSON object must contain exactly all `TrainConfig` keys (no missing keys, no extra keys).
- Resume mode in JSON uses the same `resume_checkpoint_tag` values: `auto`, `latest`, `none`, or an explicit checkpoint tag.

Padding note:

- `pad_token_id` is no longer passed via CLI or JSON config.
- The trainer reads `tokenizer.pad_token_id` from the tokenizer loaded by `model_name_or_path` and asserts it is defined.

## 6) Outputs to Watch

- Train logs: `output_dir/train_metrics.jsonl`
- Latest pointer: `output_dir/latest_checkpoint.txt`
- Periodic checkpoints: `output_dir/global_step_*`
- Final checkpoint: `output_dir/final`
