from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import random

import numpy as np
import torch

from .batch_dataset import ResolvedTrainingBatch, load_resolved_training_batches
from .collator import collate_training_samples
from .losses import compute_advantage_weighted_causal_lm_loss


@dataclass(frozen=True)
class TrainConfig:
    model_name_or_path: str
    tokenized_sqlite_path: str
    batch_sqlite_path: str
    deepspeed_config_path: str
    output_dir: str
    pad_token_id: int
    advantage_clip: float
    learning_rate: float
    weight_decay: float
    num_epochs: int
    grad_accum_steps: int
    log_interval_steps: int
    save_interval_steps: int
    seed: int


def _set_seed(seed: int) -> None:
    assert seed >= 0, "seed must be non-negative"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _is_primary_rank() -> bool:
    return (not torch.distributed.is_available()) or (
        not torch.distributed.is_initialized()
    ) or torch.distributed.get_rank() == 0


def _log_json_line(log_path: Path, payload: dict[str, float | int]) -> None:
    assert log_path.parent.exists(), f"log directory must exist: {log_path.parent}"
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def _forward_logits(model_engine: torch.nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    outputs = model_engine(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    assert hasattr(outputs, "logits"), "model forward output must contain logits"
    logits = outputs.logits
    assert isinstance(logits, torch.Tensor), "logits must be a tensor"
    return logits


def _verify_tokenizer_model_match(
    model_name_or_path: str,
    tokenizer_name_or_path: str,
    ordered_batches: list[ResolvedTrainingBatch],
    model_vocab_size: int,
) -> dict[str, str | int]:
    assert len(ordered_batches) > 0, "ordered_batches must be non-empty"
    assert model_vocab_size > 0, "model_vocab_size must be positive"

    expected_model_name = model_name_or_path.strip()
    tokenizer_name = tokenizer_name_or_path.strip()
    assert len(expected_model_name) > 0, "model_name_or_path cannot be empty"
    assert len(tokenizer_name) > 0, "tokenizer_name_or_path cannot be empty"
    assert (
        tokenizer_name == expected_model_name
    ), "tokenizer name_or_path must exactly match model_name_or_path"

    data_model_names: set[str] = set()
    max_input_token_id = -1
    max_label_token_id = -1
    for resolved_batch in ordered_batches:
        data_model_names.add(resolved_batch.model_official_name)
        for sample in resolved_batch.samples:
            data_model_names.add(sample.model_official_name)
            for token_id in sample.input_ids:
                assert token_id >= 0, "input_ids must be non-negative"
                if token_id > max_input_token_id:
                    max_input_token_id = token_id
            for token_id in sample.labels:
                if token_id == -100:
                    continue
                assert token_id >= 0, "supervised label token ids must be non-negative"
                if token_id > max_label_token_id:
                    max_label_token_id = token_id

    assert len(data_model_names) == 1, "training data must contain exactly one model_official_name"
    data_model_name = next(iter(data_model_names))
    assert (
        data_model_name == expected_model_name
    ), "training data model_official_name must match model_name_or_path"
    assert max_input_token_id < model_vocab_size, "input_ids contain token id out of model vocab range"
    assert max_label_token_id < model_vocab_size, "labels contain token id out of model vocab range"

    return {
        "model_official_name": data_model_name,
        "tokenizer_name_or_path": tokenizer_name,
        "model_vocab_size": model_vocab_size,
        "max_input_token_id": max_input_token_id,
        "max_label_token_id": max_label_token_id,
    }


def train_with_deepspeed(config: TrainConfig) -> None:
    assert config.pad_token_id >= 0, "pad_token_id must be non-negative"
    assert config.advantage_clip > 0.0, "advantage_clip must be positive"
    assert config.learning_rate > 0.0, "learning_rate must be positive"
    assert config.weight_decay >= 0.0, "weight_decay must be non-negative"
    assert config.num_epochs > 0, "num_epochs must be positive"
    assert config.grad_accum_steps > 0, "grad_accum_steps must be positive"
    assert config.log_interval_steps > 0, "log_interval_steps must be positive"
    assert config.save_interval_steps > 0, "save_interval_steps must be positive"

    import deepspeed
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _set_seed(config.seed)
    deepspeed.init_distributed()

    ordered_batches: list[ResolvedTrainingBatch] = load_resolved_training_batches(
        tokenized_sqlite_path=config.tokenized_sqlite_path,
        batch_sqlite_path=config.batch_sqlite_path,
    )

    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name_or_path,
        torch_dtype=torch.bfloat16,
    )
    model.gradient_checkpointing_enable()

    input_embeddings = model.get_input_embeddings()
    assert input_embeddings is not None, "model must expose input embeddings"
    model_vocab_size = input_embeddings.num_embeddings
    verification = _verify_tokenizer_model_match(
        model_name_or_path=config.model_name_or_path,
        tokenizer_name_or_path=tokenizer.name_or_path,
        ordered_batches=ordered_batches,
        model_vocab_size=model_vocab_size,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.95),
    )

    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        optimizer=optimizer,
        config=config.deepspeed_config_path,
    )
    assert optimizer is not None, "deepspeed must return an optimizer"

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_path = output_dir / "train_metrics.jsonl"

    if _is_primary_rank():
        _log_json_line(
            logs_path,
            {
                "step": 0,
                "epoch": -1,
                "batch_index": -1,
                "model_vocab_size": int(verification["model_vocab_size"]),
                "max_input_token_id": int(verification["max_input_token_id"]),
                "max_label_token_id": int(verification["max_label_token_id"]),
            },
        )

    global_step = 0
    accumulation_step = 0

    for epoch_index in range(config.num_epochs):
        for resolved_batch in ordered_batches:
            collated = collate_training_samples(
                samples=resolved_batch.samples,
                pad_token_id=config.pad_token_id,
            )

            device = model_engine.device
            input_ids = collated.input_ids.to(device=device, non_blocking=True)
            labels = collated.labels.to(device=device, non_blocking=True)
            attention_mask = collated.attention_mask.to(device=device, non_blocking=True)
            advantages = collated.advantages.to(device=device, non_blocking=True)

            logits = _forward_logits(model_engine, input_ids=input_ids, attention_mask=attention_mask)
            loss_output = compute_advantage_weighted_causal_lm_loss(
                logits=logits,
                labels=labels,
                advantages=advantages,
                advantage_clip=config.advantage_clip,
            )

            loss = loss_output.loss / config.grad_accum_steps
            model_engine.backward(loss)
            accumulation_step += 1

            if accumulation_step == config.grad_accum_steps:
                model_engine.step()
                accumulation_step = 0
                global_step += 1

                if _is_primary_rank() and global_step % config.log_interval_steps == 0:
                    log_payload: dict[str, float | int] = {
                        "step": global_step,
                        "epoch": epoch_index,
                        "batch_index": resolved_batch.batch_index,
                    }
                    for key, value in loss_output.stats.items():
                        log_payload[key] = value
                    _log_json_line(logs_path, log_payload)

                if global_step % config.save_interval_steps == 0:
                    checkpoint_tag = f"global_step_{global_step}"
                    model_engine.save_checkpoint(config.output_dir, tag=checkpoint_tag)

    if accumulation_step > 0:
        model_engine.step()
        global_step += 1

    model_engine.save_checkpoint(config.output_dir, tag="final")
