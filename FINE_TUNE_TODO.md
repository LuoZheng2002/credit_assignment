# Fine-Tune Framework TODO (DeepSpeed ZeRO-3)

## Scope
- Build a Python fine-tuning framework in `src_py/` for 7B/4B models on `4 x A100`.
- Train from tokenized samples with fields matching `TrainingSampleTokenized` (`input_ids`, `labels`, `advantage`).
- Use DeepSpeed ZeRO-3 with stable, reproducible training and resumable checkpoints.
- Current target model family: Qwen (`qwen2.5-7b`, `qwen3-4b`, `qwen3.5-4b`); account for potential tokenizer differences across these models.

## GPU Utilization Plan
- Current plan: LoRA adapter training (`lora`) so each rank can hold model + gradients + optimizer state on a single A100 and consume different data shards.
- Backup plan: full-model training with FSDP (`fsdp`) when LoRA quality is insufficient.
- Codebase requirement: keep shared data/loss/tokenizer-verification path and isolate plan-specific wrapping/checkpoint behavior for quick switching.

## Best-Practice Decisions
- Keep tokenizer/model identity strict so pre-tokenized IDs stay valid.
- Use per-sample weighted loss (masked CE per sample, then multiply by normalized/clipped advantage).
- Normalize and clip advantages for stability; log both raw and normalized advantage statistics.
- Prefer BF16 + ZeRO-3 + gradient checkpointing; avoid CPU offload unless memory forces it.
- Start with dynamic per-batch padding (no sequence packing in v1).
- Save resumable checkpoints (model/optimizer/scheduler/RNG/dataloader state).

## Objective (v1)
- For each sample `i`, compute:
  - `L_i = mean_t CE(logits_i,t, labels_i,t)` over tokens where `labels_i,t != -100`.
- Normalize/clip advantage:
  - `a_i' = clip((a_i - mu) / sigma, -c, c)`.
- Batch loss:
  - `L = mean_i (a_i' * L_i)`.
- Also log unweighted CE `mean_i(L_i)` for sanity.

## Implementation Plan

### 1) Project scaffolding
- [x] Create `src_py/train/` package and module layout.
- [x] Add a CLI entrypoint in `src_py/train/main.py`.
- [x] Add smoke-test launcher script `scripts/train/smoke_test_lora_qwen25.sh`.

### 2) DeepSpeed configs
- [x] Add `train_config/ds_zero3_7b.json`.
- [x] Add `train_config/ds_zero3_4b.json`.
- [x] Configure ZeRO stage 3, bf16, grad clipping, optimizer/scheduler defaults.

### 3) Data ingestion from sqlite
- [x] Implement `src_py/train/data_sqlite.py` to read `store_entries.payload_json`.
- [x] Parse to a strict sample schema (`id`, `input_ids`, `labels`, `advantage`).
- [ ] Add deterministic train/val split and shuffle with fixed seed.

### 4) Batching and collation
- [x] Implement `src_py/train/collator.py`.
- [x] Pad `input_ids` with `pad_token_id`.
- [x] Pad `labels` with `-100`.
- [x] Build `attention_mask` and include `advantages` tensor.

### 5) Loss and weighting
- [x] Implement `src_py/train/losses.py` for masked per-sample CE.
- [x] Add advantage normalization + clipping.
- [x] Compute weighted batch loss and assert finite loss.

### 6) Training engine
- [x] Implement `src_py/train/engine.py` with DeepSpeed initialize.
- [x] Load HF causal LM + tokenizer and assert `model_official_name` match with sqlite data.
- [x] Enable gradient checkpointing.
- [ ] Implement train/eval loops with distributed metric reduction.

### 12) Plan switching architecture
- [x] Add explicit training plan switch: `lora` and `fsdp`.
- [x] Keep shared pipeline for sqlite loading, collator, loss, and tokenizer verification.
- [x] Implement LoRA path with distributed batch sharding across ranks.
- [x] Implement backup FSDP path with full-model wrapping.
- [x] Keep checkpoint API compatible across both plans via a shared save helper.

### 7) Checkpointing and resume
- [ ] Save periodic DeepSpeed checkpoints.
- [ ] Support full resume (optimizer/scheduler/RNG/dataloader progression).
- [ ] Optionally export merged HF checkpoint for inference.

### 8) Metrics and logging
- [ ] Implement `src_py/train/metrics.py`.
- [ ] Log: weighted loss, unweighted CE, grad norm, tokens/sec, advantage stats.
- [ ] Add eval metrics (e.g., perplexity on masked labels).

### 9) Validation and tests
- [x] Add unit tests for sqlite parsing and schema validation.
- [x] Add unit tests for collator masks/padding behavior.
- [x] Add toy tests for loss weighting correctness.

### 11) Batch-linked loading
- [x] Implement batch-to-tokenized ID resolution for predetermined sqlite batches.
- [x] Validate all batch IDs resolve to tokenized samples.
- [x] Validate `model_official_name` consistency between tokenized samples and training batches.

### 10) Initial hyperparameter baseline
- [ ] 7B: micro-batch/GPU = 1; 4B: micro-batch/GPU = 2.
- [ ] Set grad accumulation to reach stable global supervised-token batch.
- [ ] Start with LR `1e-5`, weight decay `0.1`, betas `(0.9, 0.95)`, warmup `3%`, grad clip `1.0`.
- [ ] Start with advantage clip `3.0`.

## Risks to monitor
- Advantage scale drift causing unstable updates.
- Mismatch between tokenizer IDs and model vocabulary/revision.
- Throughput collapse from aggressive ZeRO/offload settings.
- Negative-advantage dominance early in training.

## SQLite Interaction Notes (from Rust source)
- `SqliteStore` uses table `store_entries` with schema: `id TEXT PRIMARY KEY`, `payload_json TEXT NOT NULL`.
- Tokenized sample DB file path pattern: `results/{model}/agent/{dataset}_training_tokenized_{num_samples}_{hyper_hash}.sqlite`.
- Tokenized payload schema (`TrainingSampleTokenized`): `id`, `input_ids`, `labels`, `reconstructed`, `input_length`, `advantage`.
- Batch DB file path pattern: `results/{model}/agent/{dataset}_training_batch_{num_samples}_{hyper_hash}_bs{batch_size}.sqlite`.
- Batch payload schema (`TrainingBatch`): `ids` (`QuestionNodeId[]`), `max_advantage`, `min_advantage`, `max_length`, `min_length`.
- Batch IDs are written as incremental numeric keys (`0..N-1`) via `store.upsert(batch_index, batch)`.
- Rust store scan query reads linearly with `ORDER BY id ASC`; Python reader must also read with `ORDER BY` before line-by-line iteration.
- Because `id` is TEXT, batch scanning should order numerically using `ORDER BY CAST(id AS INTEGER) ASC` for deterministic true batch order.

## SQLite Reader Tasks
- [x] Implement tokenized sample reader: `SELECT payload_json FROM store_entries ORDER BY id ASC`.
- [x] Implement batch reader: `SELECT id, payload_json FROM store_entries ORDER BY CAST(id AS INTEGER) ASC`.
- [x] Validate monotonic batch index sequence and fail fast on gaps/non-numeric IDs.

## Deliverable checklist
- [ ] End-to-end 4-GPU DeepSpeed run completes and checkpoints.
- [ ] Loss curves and advantage stats are logged and interpretable.
- [ ] Resume from checkpoint reproduces continued training behavior.
- [ ] A merged inference checkpoint can be exported.
