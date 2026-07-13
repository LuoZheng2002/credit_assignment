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

- `lora`: default plan (LoRA adapters).
- `fsdp`: full-model FSDP plan (no LoRA).

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

## 3) Epoch and Resume Behavior

Epoch definition:

- One training program run corresponds to one epoch.
- Within that run, `num_iterations` controls how many passes are made over that epoch's dataset.
- Across epochs, use a new dataset and typically a new `checkpoints_parent_dir` (for example `.../epoch_2`).

Checkpoint pointer file:

- `checkpoints_parent_dir/latest_checkpoint.txt`

The checkpoint payload folder for a run is:

- `checkpoints_parent_dir/checkpoints/`

Its internal files are owned by training/resume logic and treated as opaque externally. Current files include:

- `model_state.pt` (LoRA adapter state dict for `lora`; full model state dict for `fsdp`)
- `optimizer_state.rank{rank}.pt`
- `training_state.rank{rank}.pt`

Resume mode: always resume from the latest checkpoint if a pointer file exists, otherwise start fresh.


## 4) TOML Config Entry details

You can keep all training fields in a TOML file and launch with a minimal command.

Example config file:

- `train_request.toml` inside a job folder

Launch with script:

```bash
bash scripts/train/smoke_test_lora.sh 1 /path/to/job_folder
```

Direct `torchrun` with a job folder:

```bash
MASTER_PORT=29501 torchrun --nproc_per_node 1 --master_port "${MASTER_PORT}" -m src_py.train.main \
  --job-folder-path /path/to/job_folder
```

Job folder rules:

- The job folder must contain `train_request.toml` at its root.
- The job folder must contain `input/training_trajectories.msgpack`.
- The TOML root table must contain exactly all `TrainConfig` keys (no missing keys, no extra keys).
- Use `num_iterations` for the number of passes over loaded training batches.
- Optional sample limit for smoke tests: set `first_n_training_samples > 0` to cap visible trajectories; `0` means no limit.

Padding note:

- `pad_token_id` is no longer passed via CLI or TOML config.
- The trainer reads `tokenizer.pad_token_id` from the tokenizer loaded by `model_path` and asserts it is defined.
- `model_path` must point to a local Hugging Face model folder with safetensors weights (for example, `.../model_qwen35_08b`).

## 5) Outputs to Watch

- Train logs: `checkpoints_parent_dir/train_metrics.jsonl`
- Latest pointer: `checkpoints_parent_dir/latest_checkpoint.txt`
- Checkpoint payload (per run/epoch): `checkpoints_parent_dir/checkpoints`
- Final exported model folder: `final_model_output_parent_dir/model` (Hugging Face Transformers format with safetensors and tokenizer files)
