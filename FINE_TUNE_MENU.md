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

## 2) Start Training (JSON config path only)

Only JSON-config launch is supported now.

Use either wrapper below; both take exactly one argument:

- `scripts/launch_4gpu.sh <config_json_path>`
- `src_py/scripts/launch_4gpu.sh <config_json_path>`

Example:

```bash
bash scripts/launch_4gpu.sh src_py/configs/train_lora_example.json
```

Master port:

- Default is `29501`.
- Override with environment variable `MASTER_PORT`.

Example with custom port:

```bash
MASTER_PORT=29502 bash scripts/launch_4gpu.sh src_py/configs/train_lora_example.json
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
bash scripts/launch_4gpu.sh src_py/configs/train_lora_example.json
```

## 4) JSON Config Entry details

You can keep all training fields in a JSON file and launch with a minimal command.

Example config file:

- `src_py/configs/train_lora_example.json`

Launch with JSON script:

```bash
bash src_py/scripts/launch_4gpu.sh src_py/configs/train_lora_example.json
```

Direct `torchrun` with JSON:

```bash
MASTER_PORT=29501 torchrun --nproc_per_node 4 --master_port "${MASTER_PORT}" src_py/train/main_from_config.py \
  --config-json-path src_py/configs/train_lora_example.json
```

JSON schema rule:

- The JSON object must contain exactly all `TrainConfig` keys (no missing keys, no extra keys).
- Resume mode in JSON uses the same `resume_checkpoint_tag` values: `auto`, `latest`, `none`, or an explicit checkpoint tag.

Padding note:

- `pad_token_id` is no longer passed via CLI or JSON config.
- The trainer reads `tokenizer.pad_token_id` from the tokenizer loaded by `model_name_or_path` and asserts it is defined.

## 5) Outputs to Watch

- Train logs: `output_dir/train_metrics.jsonl`
- Latest pointer: `output_dir/latest_checkpoint.txt`
- Periodic checkpoints: `output_dir/global_step_*`
- Final checkpoint: `output_dir/final`
