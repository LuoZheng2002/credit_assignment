from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from contextlib import nullcontext
import json
import os
import random
import time

import numpy as np
import torch

from .batch_dataset import LazyResolvedBatchLoader, ResolvedTrainingBatch, load_resolved_training_batches
from .collator import collate_training_samples
from .losses import compute_advantage_weighted_causal_lm_loss


@dataclass(frozen=True)
class TrainConfig:
    training_plan: str
    model_path: str
    training_trajectory_sqlite_path: str
    checkpoints_parent_dir: str
    final_model_output_parent_dir: str
    advantage_clip: float
    learning_rate: float
    weight_decay: float
    num_iterations: int
    grad_accum_steps: int
    log_interval_steps: int
    save_interval_steps: int
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    lora_target_modules_csv: str
    resume_checkpoint_tag: str
    seed: int
    first_n_training_samples: int


@dataclass(frozen=True)
class ResumeState:
    global_step: int
    next_iteration_index: int
    next_batch_cursor: int
    accumulation_step: int
    next_sample_index: int = 0
    next_batch_size: int = 0
    adaptive_velocity: float = 0.0
    adaptive_throughput_ema: float = 0.0
    adaptive_best_throughput_ema: float = 0.0


@dataclass(frozen=True)
class AdaptiveBatchState:
    next_batch_size: int
    velocity: float
    throughput_ema: float
    best_throughput_ema: float


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


def _get_rank_world_size() -> tuple[int, int]:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(), torch.distributed.get_world_size()
    return 0, 1


def _init_distributed_device() -> torch.device:
    local_rank_env = os.environ.get("LOCAL_RANK")
    if local_rank_env is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    local_rank = int(local_rank_env)
    assert torch.cuda.is_available(), "LOCAL_RANK is set but CUDA is unavailable"
    assert local_rank >= 0, "LOCAL_RANK must be non-negative"
    torch.cuda.set_device(local_rank)

    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl")

    return torch.device("cuda", local_rank)


def _shard_batches_for_rank(
    ordered_batches: list[ResolvedTrainingBatch],
    rank: int,
    world_size: int,
) -> list[ResolvedTrainingBatch]:
    assert len(ordered_batches) > 0, "ordered_batches must be non-empty"
    assert rank >= 0, "rank must be non-negative"
    assert world_size > 0, "world_size must be positive"
    assert rank < world_size, "rank must be < world_size"

    local_batches = [batch for index, batch in enumerate(ordered_batches) if index % world_size == rank]
    assert len(local_batches) > 0, "each rank must receive at least one batch"
    return local_batches


def _save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    output_dir: Path,
    checkpoint_tag: str,
    training_plan: str,
    global_step: int,
    next_iteration_index: int,
    next_batch_cursor: int,
    accumulation_step: int,
    next_sample_index: int = 0,
    next_batch_size: int = 0,
    adaptive_velocity: float = 0.0,
    adaptive_throughput_ema: float = 0.0,
    adaptive_best_throughput_ema: float = 0.0,
) -> None:
    assert global_step >= 0, "global_step must be non-negative"
    assert next_iteration_index >= 0, "next_iteration_index must be non-negative"
    assert next_batch_cursor >= 0, "next_batch_cursor must be non-negative"
    assert accumulation_step == 0, "checkpointing with partial gradient accumulation is not supported"
    assert next_sample_index >= 0, "next_sample_index must be non-negative"
    assert next_batch_size >= 0, "next_batch_size must be non-negative"

    rank, _ = _get_rank_world_size()
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metadata_payload = {
        "global_step": global_step,
        "next_iteration_index": next_iteration_index,
        "next_batch_cursor": next_batch_cursor,
        "next_sample_index": next_sample_index,
        "next_batch_size": next_batch_size,
        "adaptive_velocity": adaptive_velocity,
        "adaptive_throughput_ema": adaptive_throughput_ema,
        "adaptive_best_throughput_ema": adaptive_best_throughput_ema,
        "accumulation_step": accumulation_step,
        "training_plan": training_plan,
        "rank": rank,
        "checkpoint_tag": checkpoint_tag,
    }
    torch.save(metadata_payload, checkpoint_dir / f"training_state.rank{rank}.pt")
    torch.save(optimizer.state_dict(), checkpoint_dir / f"optimizer_state.rank{rank}.pt")

    if training_plan == "lora_current":
        if rank == 0:
            unwrapped = model.module if hasattr(model, "module") else model
            torch.save(_extract_lora_checkpoint_state_dict(unwrapped), checkpoint_dir / "model_state.pt")
            _write_latest_checkpoint_pointer(output_dir=output_dir, checkpoint_tag=checkpoint_tag)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        return

    assert training_plan == "full_fsdp_backup", "unknown training plan for checkpointing"
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        FullStateDictConfig,
        StateDictType,
    )

    assert isinstance(model, FSDP), "full_fsdp_backup checkpoint expects FSDP model"
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
        state_dict = model.state_dict()
    if rank == 0:
        torch.save(state_dict, checkpoint_dir / "model_state.pt")
        _write_latest_checkpoint_pointer(output_dir=output_dir, checkpoint_tag=checkpoint_tag)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def _write_latest_checkpoint_pointer(output_dir: Path, checkpoint_tag: str) -> None:
    assert output_dir.exists(), f"output_dir must exist: {output_dir}"
    assert len(checkpoint_tag.strip()) > 0, "checkpoint_tag cannot be empty"
    latest_path = output_dir / "latest_checkpoint.txt"
    latest_path.write_text(checkpoint_tag.strip() + "\n", encoding="utf-8")


def _save_final_model_folder(
    model: torch.nn.Module,
    training_plan: str,
    final_model_output_parent_dir: Path,
    source_model_path: str,
    tokenizer: object,
) -> None:
    final_model_output_path = final_model_output_parent_dir / "model"
    rank, _ = _get_rank_world_size()
    if rank == 0:
        if final_model_output_parent_dir.exists():
            assert final_model_output_parent_dir.is_dir(), (
                "final_model_output_parent_dir must be a directory when it exists: "
                f"{final_model_output_parent_dir}"
            )
        final_model_output_parent_dir.mkdir(parents=True, exist_ok=True)
        if final_model_output_path.exists():
            assert final_model_output_path.is_dir(), (
                f"final_model_output_path must be a directory when it exists: {final_model_output_path}"
            )
        final_model_output_path.mkdir(parents=True, exist_ok=True)

    if training_plan == "lora_current":
        if rank == 0:
            unwrapped = model.module if hasattr(model, "module") else model
            export_model = unwrapped.merge_and_unload() if hasattr(unwrapped, "merge_and_unload") else unwrapped
            export_model.save_pretrained(final_model_output_path, safe_serialization=True)
            if hasattr(tokenizer, "save_pretrained"):
                tokenizer.save_pretrained(final_model_output_path)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        return

    assert training_plan == "full_fsdp_backup", "unknown training plan for final model export"
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        FullStateDictConfig,
        StateDictType,
    )

    assert isinstance(model, FSDP), "full_fsdp_backup final export expects FSDP model"
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
        state_dict = model.state_dict()
    if rank == 0:
        from transformers import AutoModelForCausalLM

        export_model = AutoModelForCausalLM.from_pretrained(source_model_path, torch_dtype=torch.bfloat16)
        incompatible = export_model.load_state_dict(state_dict, strict=True)
        assert len(incompatible.missing_keys) == 0, "final export state_dict is missing keys"
        assert len(incompatible.unexpected_keys) == 0, "final export state_dict has unexpected keys"
        export_model.save_pretrained(final_model_output_path, safe_serialization=True)
        if hasattr(tokenizer, "save_pretrained"):
            tokenizer.save_pretrained(final_model_output_path)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def _read_latest_checkpoint_pointer(output_dir: Path) -> str:
    latest_path = output_dir / "latest_checkpoint.txt"
    assert latest_path.exists(), f"latest checkpoint pointer not found: {latest_path}"
    checkpoint_tag = latest_path.read_text(encoding="utf-8").strip()
    assert len(checkpoint_tag) > 0, f"latest checkpoint pointer is empty: {latest_path}"
    return checkpoint_tag


def _resolve_resume_checkpoint_tag(output_dir: Path, resume_checkpoint_tag: str) -> str:
    normalized_tag = resume_checkpoint_tag.strip()
    assert len(normalized_tag) > 0, "resume_checkpoint_tag cannot be empty"
    if normalized_tag == "none":
        return ""
    if normalized_tag in {"latest", "auto"}:
        latest_path = output_dir / "latest_checkpoint.txt"
        if not latest_path.exists():
            if normalized_tag == "auto":
                return ""
            assert latest_path.exists(), f"latest checkpoint pointer not found: {latest_path}"
        return _read_latest_checkpoint_pointer(output_dir=output_dir)
    assert normalized_tag == "checkpoints", (
        "explicit resume_checkpoint_tag must be 'checkpoints' for single-epoch run layout"
    )
    return normalized_tag


def _extract_lora_checkpoint_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    if hasattr(model, "peft_config"):
        try:
            from peft import get_peft_model_state_dict

            return get_peft_model_state_dict(model)
        except ImportError:
            pass
    return model.state_dict()


def _load_lora_checkpoint_state_dict(model: torch.nn.Module, state_dict: dict[str, torch.Tensor]) -> None:
    if hasattr(model, "peft_config"):
        try:
            from peft import set_peft_model_state_dict

            set_peft_model_state_dict(model, state_dict)
            return
        except ImportError:
            pass

    incompatible = model.load_state_dict(state_dict, strict=True)
    assert len(incompatible.missing_keys) == 0, "checkpoint model state is missing keys"
    assert len(incompatible.unexpected_keys) == 0, "checkpoint model state has unexpected keys"


def _load_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    output_dir: Path,
    checkpoint_tag: str,
    training_plan: str,
) -> ResumeState:
    assert len(checkpoint_tag.strip()) > 0, "checkpoint_tag cannot be empty"

    rank, _ = _get_rank_world_size()
    checkpoint_dir = output_dir / "checkpoints"
    assert checkpoint_dir.exists(), f"checkpoint directory not found: {checkpoint_dir}"
    model_state_path = checkpoint_dir / "model_state.pt"
    optimizer_state_path = checkpoint_dir / f"optimizer_state.rank{rank}.pt"
    training_state_path = checkpoint_dir / f"training_state.rank{rank}.pt"

    assert model_state_path.exists(), f"missing model state: {model_state_path}"
    assert optimizer_state_path.exists(), f"missing optimizer state: {optimizer_state_path}"
    assert training_state_path.exists(), f"missing training state: {training_state_path}"

    model_state_dict = torch.load(model_state_path, map_location="cpu")
    if training_plan == "lora_current":
        unwrapped = model.module if hasattr(model, "module") else model
        assert isinstance(model_state_dict, dict), "checkpoint model state must be a state_dict"
        _load_lora_checkpoint_state_dict(unwrapped, model_state_dict)
    else:
        assert training_plan == "full_fsdp_backup", "unknown training plan for checkpoint loading"
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP,
            FullStateDictConfig,
            StateDictType,
        )

        assert isinstance(model, FSDP), "full_fsdp_backup loading expects FSDP model"
        load_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=False)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, load_policy):
            incompatible = model.load_state_dict(model_state_dict, strict=True)

    if training_plan != "lora_current":
        assert len(incompatible.missing_keys) == 0, "checkpoint model state is missing keys"
        assert len(incompatible.unexpected_keys) == 0, "checkpoint model state has unexpected keys"

    optimizer_state = torch.load(optimizer_state_path, map_location="cpu")
    optimizer.load_state_dict(optimizer_state)

    training_state = torch.load(training_state_path, map_location="cpu")
    assert training_state["training_plan"] == training_plan, "checkpoint training plan mismatch"
    next_iteration_index_obj = training_state.get("next_iteration_index")
    if next_iteration_index_obj is None:
        next_iteration_index_obj = training_state.get("next_epoch_index")
    assert next_iteration_index_obj is not None, "checkpoint missing next_iteration_index"
    resume_state = ResumeState(
        global_step=int(training_state["global_step"]),
        next_iteration_index=int(next_iteration_index_obj),
        next_batch_cursor=int(training_state["next_batch_cursor"]),
        accumulation_step=int(training_state["accumulation_step"]),
        next_sample_index=int(training_state.get("next_sample_index", 0)),
        next_batch_size=int(training_state.get("next_batch_size", 0)),
        adaptive_velocity=float(training_state.get("adaptive_velocity", 0.0)),
        adaptive_throughput_ema=float(training_state.get("adaptive_throughput_ema", 0.0)),
        adaptive_best_throughput_ema=float(training_state.get("adaptive_best_throughput_ema", 0.0)),
    )
    assert resume_state.global_step >= 0, "resume global_step must be non-negative"
    assert resume_state.next_iteration_index >= 0, "resume iteration index must be non-negative"
    assert resume_state.next_batch_cursor >= 0, "resume batch cursor must be non-negative"
    assert resume_state.next_sample_index >= 0, "resume sample index must be non-negative"
    assert resume_state.next_batch_size >= 0, "resume next_batch_size must be non-negative"
    assert np.isfinite(resume_state.adaptive_velocity), "resume adaptive_velocity must be finite"
    assert np.isfinite(resume_state.adaptive_throughput_ema), "resume adaptive_throughput_ema must be finite"
    assert np.isfinite(
        resume_state.adaptive_best_throughput_ema
    ), "resume adaptive_best_throughput_ema must be finite"
    assert (
        resume_state.accumulation_step == 0
    ), "resuming from partial gradient accumulation is not supported"
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()
    return resume_state


def _compute_next_position(
    iteration_index: int,
    local_batch_cursor: int,
    local_batch_count: int,
) -> tuple[int, int]:
    assert iteration_index >= 0, "iteration_index must be non-negative"
    assert local_batch_cursor >= 0, "local_batch_cursor must be non-negative"
    assert local_batch_count > 0, "local_batch_count must be positive"
    assert local_batch_cursor < local_batch_count, "local_batch_cursor must be in range"

    next_iteration_index = iteration_index
    next_batch_cursor = local_batch_cursor + 1
    if next_batch_cursor == local_batch_count:
        next_iteration_index += 1
        next_batch_cursor = 0
    return next_iteration_index, next_batch_cursor


def _update_adaptive_batch_state(
    adaptive_state: AdaptiveBatchState,
    measured_throughput: float,
    min_batch_size: int,
    max_batch_size: int,
) -> AdaptiveBatchState:
    assert measured_throughput > 0.0, "measured_throughput must be positive"
    assert min_batch_size > 0, "min_batch_size must be positive"
    assert max_batch_size >= min_batch_size, "max_batch_size must be >= min_batch_size"

    ema_alpha = 0.2
    momentum = 0.8
    min_improvement_ratio = 0.01
    base_step = 0.08
    max_velocity = 0.35

    updated_ema = adaptive_state.throughput_ema
    if updated_ema <= 0.0:
        updated_ema = measured_throughput
    else:
        updated_ema = (1.0 - ema_alpha) * adaptive_state.throughput_ema + ema_alpha * measured_throughput

    previous_best_ema = adaptive_state.best_throughput_ema
    best_ema = max(previous_best_ema, updated_ema)
    improvement_ratio = 0.0
    if previous_best_ema > 0.0:
        improvement_ratio = (updated_ema - previous_best_ema) / previous_best_ema

    direction = -1.0
    if improvement_ratio > -min_improvement_ratio:
        direction = 1.0

    velocity = momentum * adaptive_state.velocity + direction * base_step
    velocity = min(max_velocity, max(-max_velocity, velocity))

    candidate_batch_size = int(round(adaptive_state.next_batch_size * (1.0 + velocity)))
    if candidate_batch_size == adaptive_state.next_batch_size:
        if velocity > 0.0:
            candidate_batch_size += 1
        elif velocity < 0.0:
            candidate_batch_size -= 1
    candidate_batch_size = max(min_batch_size, min(max_batch_size, candidate_batch_size))

    if candidate_batch_size == max_batch_size and velocity > 0.0:
        velocity = 0.0
    if candidate_batch_size == min_batch_size and velocity < 0.0:
        velocity = 0.0

    return AdaptiveBatchState(
        next_batch_size=candidate_batch_size,
        velocity=velocity,
        throughput_ema=updated_ema,
        best_throughput_ema=max(best_ema, updated_ema),
    )


def _resolve_pad_token_id(tokenizer_pad_token_id: int | None) -> int:
    assert tokenizer_pad_token_id is not None, "tokenizer.pad_token_id must be defined for training"
    assert tokenizer_pad_token_id >= 0, "tokenizer.pad_token_id must be non-negative"
    return int(tokenizer_pad_token_id)


def _normalize_optional_token_id(token_id: int | None) -> int:
    if token_id is None:
        return -1
    assert token_id >= 0, "token id must be non-negative when defined"
    return int(token_id)


def _resolve_local_model_path(model_path: str) -> str:
    normalized = Path(model_path).expanduser().resolve()
    assert normalized.exists(), f"model_path does not exist: {normalized}"
    assert normalized.is_dir(), f"model_path must be a directory: {normalized}"

    required_files = [
        normalized / "config.json",
        normalized / "tokenizer_config.json",
    ]
    for required_file in required_files:
        assert required_file.is_file(), f"missing required model file: {required_file}"

    has_safetensors_weights = (normalized / "model.safetensors").is_file() or (
        normalized / "model.safetensors.index.json"
    ).is_file()
    assert has_safetensors_weights, (
        "model_path must contain safetensors weights (model.safetensors or "
        "model.safetensors.index.json)"
    )
    return str(normalized)


def _verify_tokenizer_model_match(
    model_path: str,
    tokenizer_name_or_path: str,
    ordered_batches: list[ResolvedTrainingBatch],
    model_vocab_size: int,
) -> dict[str, str | int]:
    assert len(ordered_batches) > 0, "ordered_batches must be non-empty"
    assert model_vocab_size > 0, "model_vocab_size must be positive"

    expected_model_name = model_path.strip()
    tokenizer_name = tokenizer_name_or_path.strip()
    assert len(expected_model_name) > 0, "model_path cannot be empty"
    assert len(tokenizer_name) > 0, "tokenizer_name_or_path cannot be empty"
    assert (
        tokenizer_name == expected_model_name
    ), "tokenizer name_or_path must exactly match model_path"

    data_model_names: set[str] = set()
    max_input_token_id = -1
    max_label_token_id = -1
    for resolved_batch in ordered_batches:
        if len(resolved_batch.model_official_name.strip()) > 0:
            data_model_names.add(resolved_batch.model_official_name)
        for sample in resolved_batch.samples:
            if len(sample.model_official_name.strip()) > 0:
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

    if len(data_model_names) > 0:
        assert len(data_model_names) == 1, "training data must contain exactly one model_official_name"
        data_model_name = next(iter(data_model_names))
        assert (
            data_model_name == expected_model_name
        ), "training data model_official_name must match model_path"
    else:
        data_model_name = expected_model_name
    assert max_input_token_id < model_vocab_size, "input_ids contain token id out of model vocab range"
    assert max_label_token_id < model_vocab_size, "labels contain token id out of model vocab range"

    return {
        "model_official_name": data_model_name,
        "tokenizer_name_or_path": tokenizer_name,
        "model_vocab_size": model_vocab_size,
        "max_input_token_id": max_input_token_id,
        "max_label_token_id": max_label_token_id,
    }


def _build_lora_model(
    model_path: str,
    lora_rank: int,
    lora_alpha: int,
    lora_dropout: float,
    lora_target_modules_csv: str,
    device: torch.device,
) -> torch.nn.Module:
    assert lora_rank > 0, "lora_rank must be positive"
    assert lora_alpha > 0, "lora_alpha must be positive"
    assert lora_dropout >= 0.0 and lora_dropout < 1.0, "lora_dropout must be in [0, 1)"
    targets = [value.strip() for value in lora_target_modules_csv.split(",") if value.strip()]
    assert len(targets) > 0, "lora_target_modules_csv must contain at least one module"

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
    ).to(device)
    base_model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=targets,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, lora_config)
    trainable_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    assert trainable_count > 0, "LoRA model must expose trainable parameters"
    return model


def _build_fsdp_model(model_path: str, device: torch.device) -> torch.nn.Module:
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import MixedPrecision
    from transformers import AutoModelForCausalLM

    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
    ).to(device)
    base_model.gradient_checkpointing_enable()

    mixed_precision = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )
    return FSDP(base_model, device_id=device, mixed_precision=mixed_precision)


def train_with_deepspeed(config: TrainConfig) -> None:
    assert config.training_plan in {
        "lora_current",
        "full_fsdp_backup",
    }, "training_plan must be one of: lora_current, full_fsdp_backup"
    assert config.advantage_clip > 0.0, "advantage_clip must be positive"
    assert config.learning_rate > 0.0, "learning_rate must be positive"
    assert config.weight_decay >= 0.0, "weight_decay must be non-negative"
    assert config.num_iterations > 0, "num_iterations must be positive"
    assert config.grad_accum_steps > 0, "grad_accum_steps must be positive"
    assert config.log_interval_steps > 0, "log_interval_steps must be positive"
    assert config.save_interval_steps > 0, "save_interval_steps must be positive"
    assert len(config.resume_checkpoint_tag.strip()) > 0, "resume_checkpoint_tag cannot be empty"
    assert len(config.checkpoints_parent_dir.strip()) > 0, "checkpoints_parent_dir cannot be empty"
    assert len(config.final_model_output_parent_dir.strip()) > 0, "final_model_output_parent_dir cannot be empty"
    assert config.first_n_training_samples >= 0, "first_n_training_samples must be non-negative"

    from transformers import AutoTokenizer

    _set_seed(config.seed)
    device = _init_distributed_device()
    rank, world_size = _get_rank_world_size()
    initial_batch_size = 1
    initial_adaptive_velocity = 0.12

    resolved_model_path = _resolve_local_model_path(config.model_path)
    tokenizer = AutoTokenizer.from_pretrained(resolved_model_path)
    pad_token_id = _resolve_pad_token_id(tokenizer.pad_token_id)
    eos_token_id = _normalize_optional_token_id(tokenizer.eos_token_id)
    bos_token_id = _normalize_optional_token_id(tokenizer.bos_token_id)

    if config.training_plan == "lora_current":
        model = _build_lora_model(
            model_path=resolved_model_path,
            lora_rank=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            lora_target_modules_csv=config.lora_target_modules_csv,
            device=device,
        )
    else:
        model = _build_fsdp_model(model_path=resolved_model_path, device=device)

    input_embeddings = model.get_input_embeddings()
    assert input_embeddings is not None, "model must expose input embeddings"
    model_vocab_size = input_embeddings.num_embeddings

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.95),
    )

    if config.training_plan == "lora_current" and world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device.index],
            output_device=device.index,
            find_unused_parameters=False,
        )

    checkpoints_parent_dir = Path(config.checkpoints_parent_dir)
    checkpoints_parent_dir.mkdir(parents=True, exist_ok=True)
    final_model_output_parent_dir = Path(config.final_model_output_parent_dir)
    logs_path = checkpoints_parent_dir / "train_metrics.jsonl"

    expected_model_name = resolved_model_path
    tokenizer_name = tokenizer.name_or_path.strip()
    assert len(expected_model_name) > 0, "model_path cannot be empty"
    assert len(tokenizer_name) > 0, "tokenizer_name_or_path cannot be empty"
    assert tokenizer_name == expected_model_name, "tokenizer name_or_path must exactly match model_path"

    resolved_resume_tag = _resolve_resume_checkpoint_tag(
        output_dir=checkpoints_parent_dir,
        resume_checkpoint_tag=config.resume_checkpoint_tag,
    )

    resume_state = ResumeState(
        global_step=0,
        next_iteration_index=0,
        next_batch_cursor=0,
        accumulation_step=0,
        next_sample_index=0,
        next_batch_size=initial_batch_size,
        adaptive_velocity=initial_adaptive_velocity,
    )
    if len(resolved_resume_tag) > 0:
        resume_state = _load_checkpoint(
            model=model,
            optimizer=optimizer,
            output_dir=checkpoints_parent_dir,
            checkpoint_tag=resolved_resume_tag,
            training_plan=config.training_plan,
        )
        if resume_state.next_batch_size == 0:
            resume_state = ResumeState(
                global_step=resume_state.global_step,
                next_iteration_index=resume_state.next_iteration_index,
                next_batch_cursor=resume_state.next_batch_cursor,
                accumulation_step=resume_state.accumulation_step,
                next_sample_index=resume_state.next_sample_index,
                next_batch_size=initial_batch_size,
                adaptive_velocity=initial_adaptive_velocity,
                adaptive_throughput_ema=resume_state.adaptive_throughput_ema,
                adaptive_best_throughput_ema=resume_state.adaptive_best_throughput_ema,
            )

    if world_size > 1:
        ordered_batches: list[ResolvedTrainingBatch] = load_resolved_training_batches(
            training_trajectory_sqlite_path=config.training_trajectory_sqlite_path,
            batch_size=initial_batch_size,
            model_official_name=expected_model_name,
            first_n_training_samples=config.first_n_training_samples,
        )
        local_batches = _shard_batches_for_rank(ordered_batches=ordered_batches, rank=rank, world_size=world_size)
        verification = _verify_tokenizer_model_match(
            model_path=expected_model_name,
            tokenizer_name_or_path=tokenizer.name_or_path,
            ordered_batches=ordered_batches,
            model_vocab_size=model_vocab_size,
        )

        if _is_primary_rank():
            print(
                "[startup] tokenizer special tokens "
                f"pad_token_id={pad_token_id} eos_token_id={eos_token_id} bos_token_id={bos_token_id}"
            )
            _log_json_line(
                logs_path,
                {
                    "step": 0,
                    "iteration": -1,
                    "batch_index": -1,
                    "model_vocab_size": int(verification["model_vocab_size"]),
                    "max_input_token_id": int(verification["max_input_token_id"]),
                    "max_label_token_id": int(verification["max_label_token_id"]),
                    "pad_token_id": pad_token_id,
                    "eos_token_id": eos_token_id,
                    "bos_token_id": bos_token_id,
                    "rank": rank,
                    "world_size": world_size,
                    "local_batch_count": len(local_batches),
                    "next_batch_size": initial_batch_size,
                },
            )

        assert resume_state.next_batch_cursor < len(local_batches) or (
            resume_state.next_iteration_index >= config.num_iterations and resume_state.next_batch_cursor == 0
        ), "resume batch cursor is out of local batch range"

        global_step = resume_state.global_step
        accumulation_step = resume_state.accumulation_step
        optimizer.zero_grad(set_to_none=True)

        if resume_state.next_iteration_index >= config.num_iterations:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.barrier()
            return

        for iteration_index in range(resume_state.next_iteration_index, config.num_iterations):
            batch_start_cursor = (
                resume_state.next_batch_cursor if iteration_index == resume_state.next_iteration_index else 0
            )
            for local_batch_cursor in range(batch_start_cursor, len(local_batches)):
                resolved_batch = local_batches[local_batch_cursor]
                collated = collate_training_samples(
                    samples=resolved_batch.samples,
                    pad_token_id=pad_token_id,
                )

                input_ids = collated.input_ids.to(device=device, non_blocking=True)
                labels = collated.labels.to(device=device, non_blocking=True)
                attention_mask = collated.attention_mask.to(device=device, non_blocking=True)
                advantages = collated.advantages.to(device=device, non_blocking=True)

                should_sync = (accumulation_step + 1) == config.grad_accum_steps
                sync_context = nullcontext()
                if (
                    config.training_plan == "lora_current"
                    and world_size > 1
                    and hasattr(model, "no_sync")
                    and not should_sync
                ):
                    sync_context = model.no_sync()

                with sync_context:
                    logits = _forward_logits(model, input_ids=input_ids, attention_mask=attention_mask)
                    loss_output = compute_advantage_weighted_causal_lm_loss(
                        logits=logits,
                        labels=labels,
                        advantages=advantages,
                        advantage_clip=config.advantage_clip,
                    )

                    loss = loss_output.loss / config.grad_accum_steps
                    loss.backward()

                accumulation_step += 1

                if accumulation_step == config.grad_accum_steps:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    accumulation_step = 0
                    global_step += 1

                    if _is_primary_rank() and global_step % config.log_interval_steps == 0:
                        log_payload: dict[str, float | int] = {
                            "step": global_step,
                            "iteration": iteration_index,
                            "batch_index": resolved_batch.batch_index,
                            "next_batch_size": initial_batch_size,
                        }
                        for key, value in loss_output.stats.items():
                            log_payload[key] = value
                        _log_json_line(logs_path, log_payload)

                    if global_step % config.save_interval_steps == 0:
                        next_iteration_index, next_batch_cursor = _compute_next_position(
                            iteration_index=iteration_index,
                            local_batch_cursor=local_batch_cursor,
                            local_batch_count=len(local_batches),
                        )
                        checkpoint_tag = "checkpoints"
                        _save_checkpoint(
                            model=model,
                            optimizer=optimizer,
                            output_dir=checkpoints_parent_dir,
                            checkpoint_tag=checkpoint_tag,
                            training_plan=config.training_plan,
                            global_step=global_step,
                            next_iteration_index=next_iteration_index,
                            next_batch_cursor=next_batch_cursor,
                            accumulation_step=accumulation_step,
                            next_batch_size=initial_batch_size,
                        )

        if accumulation_step > 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            accumulation_step = 0

        _save_checkpoint(
            model=model,
            optimizer=optimizer,
            output_dir=checkpoints_parent_dir,
            checkpoint_tag="checkpoints",
            training_plan=config.training_plan,
            global_step=global_step,
            next_iteration_index=config.num_iterations,
            next_batch_cursor=0,
            accumulation_step=accumulation_step,
            next_batch_size=initial_batch_size,
        )
        _save_final_model_folder(
            model=model,
            training_plan=config.training_plan,
            final_model_output_parent_dir=final_model_output_parent_dir,
            source_model_path=resolved_model_path,
            tokenizer=tokenizer,
        )

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        return

    lazy_loader = LazyResolvedBatchLoader(
        training_trajectory_sqlite_path=config.training_trajectory_sqlite_path,
        model_official_name=expected_model_name,
        first_n_training_samples=config.first_n_training_samples,
    )
    try:
        fallback_sample_index = resume_state.next_batch_cursor * initial_batch_size
        resume_sample_index = resume_state.next_sample_index
        if resume_sample_index == 0 and resume_state.next_batch_cursor > 0:
            resume_sample_index = fallback_sample_index
        assert resume_sample_index <= lazy_loader.sample_count, "resume sample index is out of range"

        adaptive_state = AdaptiveBatchState(
            next_batch_size=max(1, min(lazy_loader.sample_count, resume_state.next_batch_size)),
            velocity=(
                resume_state.adaptive_velocity
                if resume_state.adaptive_velocity > 0.0
                else initial_adaptive_velocity
            ),
            throughput_ema=resume_state.adaptive_throughput_ema,
            best_throughput_ema=resume_state.adaptive_best_throughput_ema,
        )

        max_input_token_id = -1
        max_label_token_id = -1
        data_model_name = expected_model_name

        if _is_primary_rank():
            print(
                "[startup] tokenizer special tokens "
                f"pad_token_id={pad_token_id} eos_token_id={eos_token_id} bos_token_id={bos_token_id}"
            )
            _log_json_line(
                logs_path,
                {
                    "step": 0,
                    "iteration": -1,
                    "batch_index": -1,
                    "model_vocab_size": int(model_vocab_size),
                    "max_input_token_id": -1,
                    "max_label_token_id": -1,
                    "pad_token_id": pad_token_id,
                    "eos_token_id": eos_token_id,
                    "bos_token_id": bos_token_id,
                    "rank": rank,
                    "world_size": world_size,
                    "local_batch_count": -1,
                    "next_batch_size": adaptive_state.next_batch_size,
                },
            )

        global_step = resume_state.global_step
        accumulation_step = resume_state.accumulation_step
        optimizer.zero_grad(set_to_none=True)

        if resume_state.next_iteration_index >= config.num_iterations:
            return

        for iteration_index in range(resume_state.next_iteration_index, config.num_iterations):
            sample_index = resume_sample_index if iteration_index == resume_state.next_iteration_index else 0
            batch_index = 0
            while sample_index < lazy_loader.sample_count:
                requested_batch_size = min(adaptive_state.next_batch_size, lazy_loader.sample_count - sample_index)
                if requested_batch_size <= 0:
                    break

                if _is_primary_rank():
                    print(
                        "[batch] "
                        f"iteration={iteration_index} "
                        f"batch_index={batch_index} "
                        f"requested_batch_size={requested_batch_size} "
                        f"next_batch_size={adaptive_state.next_batch_size}"
                    )

                window = lazy_loader.resolve_batch(
                    sample_index=sample_index,
                    batch_size=requested_batch_size,
                    batch_index=batch_index,
                )
                resolved_batch = window.resolved_batch

                for sample in resolved_batch.samples:
                    if len(sample.model_official_name.strip()) > 0:
                        data_model_name = sample.model_official_name.strip()
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

                assert data_model_name == expected_model_name, (
                    "training data model_official_name must match model_path"
                )
                assert max_input_token_id < model_vocab_size, "input_ids contain token id out of model vocab range"
                assert max_label_token_id < model_vocab_size, "labels contain token id out of model vocab range"

                collated = collate_training_samples(
                    samples=resolved_batch.samples,
                    pad_token_id=pad_token_id,
                )

                input_ids = collated.input_ids.to(device=device, non_blocking=True)
                labels = collated.labels.to(device=device, non_blocking=True)
                attention_mask = collated.attention_mask.to(device=device, non_blocking=True)
                advantages = collated.advantages.to(device=device, non_blocking=True)

                should_sync = (accumulation_step + 1) == config.grad_accum_steps
                sync_context = nullcontext()

                step_start = time.perf_counter()
                try:
                    with sync_context:
                        logits = _forward_logits(model, input_ids=input_ids, attention_mask=attention_mask)
                        loss_output = compute_advantage_weighted_causal_lm_loss(
                            logits=logits,
                            labels=labels,
                            advantages=advantages,
                            advantage_clip=config.advantage_clip,
                        )

                        loss = loss_output.loss / config.grad_accum_steps
                        loss.backward()
                except torch.cuda.OutOfMemoryError:
                    optimizer.zero_grad(set_to_none=True)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    if adaptive_state.next_batch_size <= 1:
                        raise
                    reduced_batch_size = max(1, adaptive_state.next_batch_size // 2)
                    adaptive_state = AdaptiveBatchState(
                        next_batch_size=reduced_batch_size,
                        velocity=0.0,
                        throughput_ema=adaptive_state.throughput_ema,
                        best_throughput_ema=adaptive_state.best_throughput_ema,
                    )
                    if _is_primary_rank():
                        _log_json_line(
                            logs_path,
                            {
                                "step": global_step,
                                "iteration": iteration_index,
                                "batch_index": batch_index,
                                "oom": 1,
                                "next_batch_size": adaptive_state.next_batch_size,
                            },
                        )
                    continue

                step_elapsed_sec = max(time.perf_counter() - step_start, 1e-6)
                throughput_samples_per_sec = float(len(resolved_batch.samples)) / step_elapsed_sec

                accumulation_step += 1
                sample_index = window.next_sample_index
                batch_index += 1

                if accumulation_step == config.grad_accum_steps:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    accumulation_step = 0
                    global_step += 1

                    adaptive_state = _update_adaptive_batch_state(
                        adaptive_state=adaptive_state,
                        measured_throughput=throughput_samples_per_sec,
                        min_batch_size=1,
                        max_batch_size=lazy_loader.sample_count,
                    )

                    if _is_primary_rank() and global_step % config.log_interval_steps == 0:
                        log_payload: dict[str, float | int] = {
                            "step": global_step,
                            "iteration": iteration_index,
                            "batch_index": resolved_batch.batch_index,
                            "next_batch_size": adaptive_state.next_batch_size,
                            "actual_batch_size": len(resolved_batch.samples),
                            "step_time_sec": float(step_elapsed_sec),
                            "throughput_samples_per_sec": throughput_samples_per_sec,
                        }
                        for key, value in loss_output.stats.items():
                            log_payload[key] = value
                        _log_json_line(logs_path, log_payload)

                    if global_step % config.save_interval_steps == 0:
                        next_iteration_index = iteration_index
                        next_sample_index = sample_index
                        if next_sample_index >= lazy_loader.sample_count:
                            next_iteration_index += 1
                            next_sample_index = 0
                        checkpoint_tag = "checkpoints"
                        _save_checkpoint(
                            model=model,
                            optimizer=optimizer,
                            output_dir=checkpoints_parent_dir,
                            checkpoint_tag=checkpoint_tag,
                            training_plan=config.training_plan,
                            global_step=global_step,
                            next_iteration_index=next_iteration_index,
                            next_batch_cursor=max(0, next_sample_index // max(1, adaptive_state.next_batch_size)),
                            accumulation_step=accumulation_step,
                            next_sample_index=next_sample_index,
                            next_batch_size=adaptive_state.next_batch_size,
                            adaptive_velocity=adaptive_state.velocity,
                            adaptive_throughput_ema=adaptive_state.throughput_ema,
                            adaptive_best_throughput_ema=adaptive_state.best_throughput_ema,
                        )
            resume_sample_index = 0

        if accumulation_step > 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            accumulation_step = 0

        _save_checkpoint(
            model=model,
            optimizer=optimizer,
            output_dir=checkpoints_parent_dir,
            checkpoint_tag="checkpoints",
            training_plan=config.training_plan,
            global_step=global_step,
            next_iteration_index=config.num_iterations,
            next_batch_cursor=0,
            accumulation_step=accumulation_step,
            next_sample_index=0,
            next_batch_size=adaptive_state.next_batch_size,
            adaptive_velocity=adaptive_state.velocity,
            adaptive_throughput_ema=adaptive_state.throughput_ema,
            adaptive_best_throughput_ema=adaptive_state.best_throughput_ema,
        )
        _save_final_model_folder(
            model=model,
            training_plan=config.training_plan,
            final_model_output_parent_dir=final_model_output_parent_dir,
            source_model_path=resolved_model_path,
            tokenizer=tokenizer,
        )
    finally:
        lazy_loader.close()

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()
