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

## 2) Start Training (TOML config path only)

Only TOML-config launch is supported now.

Use either the smoke-test helper or direct `torchrun`.

Example:

```bash
bash scripts/train/smoke_test_lora.sh 1 train_config/lora_qwen25.toml
```

Master port:

- Default is `29501`.
- Override with environment variable `MASTER_PORT`.

Example with custom port:

```bash
MASTER_PORT=29502 bash scripts/train/smoke_test_lora.sh 2 train_config/lora_qwen25.toml
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
# 1) Edit the config field "resume_checkpoint_tag"
#    auto | latest | none | global_step_100

# 2) Launch using config-path-only entry
bash scripts/train/smoke_test_lora.sh 1 train_config/lora_qwen25.toml
```

## 4) TOML Config Entry details

You can keep all training fields in a TOML file and launch with a minimal command.

Example config file:

- `train_config/lora_qwen25.toml`

Launch with JSON script:

```bash
bash scripts/train/smoke_test_lora.sh 1 train_config/lora_qwen25.toml
```

Direct `torchrun` with TOML:

```bash
MASTER_PORT=29501 torchrun --nproc_per_node 1 --master_port "${MASTER_PORT}" src_py/train/main_from_config.py \
  --config-toml-path train_config/lora_qwen25.toml
```

TOML schema rule:

- The TOML root table must contain exactly all `TrainConfig` keys (no missing keys, no extra keys).
- Resume mode in JSON uses the same `resume_checkpoint_tag` values: `auto`, `latest`, `none`, or an explicit checkpoint tag.

Padding note:

- `pad_token_id` is no longer passed via CLI or TOML config.
- The trainer reads `tokenizer.pad_token_id` from the tokenizer loaded by `model_name_or_path` and asserts it is defined.

## 5) Outputs to Watch

- Train logs: `output_dir/train_metrics.jsonl`
- Latest pointer: `output_dir/latest_checkpoint.txt`
- Periodic checkpoints: `output_dir/global_step_*`
- Final checkpoint: `output_dir/final`
