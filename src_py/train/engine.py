from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from contextlib import nullcontext
import atexit
import json
import os
import random
import shutil
import sys
import time

import numpy as np
import torch

from .batch_dataset import LazyResolvedBatchLoader, ResolvedTrainingBatch, load_resolved_training_batches
from .collator import collate_training_samples
from .losses import compute_advantage_weighted_causal_lm_loss


@dataclass(frozen=True)
class TrainConfig:
    training_plan: str
    model_parent_dir: str
    training_trajectory_sqlite_path: str
    checkpoints_parent_dir: str
    final_model_output_parent_dir: str
    advantage_clip: float
    learning_rate: float
    weight_decay: float
    num_iterations: int
    grad_accum_steps: int
    log_time_interval: float
    checkpoint_save_time_interval: float
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
    adaptive_memory_utilization_ema: float = 0.0
    adaptive_previous_tokens_per_sample: float = 0.0
    adaptive_next_batch_size_float: float = 0.0


@dataclass(frozen=True)
class AdaptiveBatchState:
    next_batch_size: int
    next_batch_size_float: float
    velocity: float
    throughput_ema: float
    best_throughput_ema: float
    memory_utilization_ema: float
    previous_tokens_per_sample: float


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


def _json_key_value(key: str, value: object) -> str:
    return json.dumps({"KeyValuePair": {"key": key, "value": str(value)}})


def _json_master_progress(progress: float, label: str) -> str:
    return json.dumps({"MasterProgress": {"progress": progress, "label": label}})


def _json_worker_progress(worker_name: str, progress: float, label: str) -> str:
    return json.dumps({"WorkerProgress": {"worker_name": worker_name, "progress": progress, "label": label}})


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


def _distributed_barrier() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def _shutdown_distributed_process_group() -> None:
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return
    try:
        torch.distributed.barrier()
    finally:
        torch.distributed.destroy_process_group()


def _is_cuda_oom_exception(exc: BaseException) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    if not isinstance(exc, RuntimeError):
        return False
    message = str(exc).lower()
    return "out of memory" in message and "cuda" in message


def _print_cuda_oom_stderr(
    *,
    rank: int,
    iteration_index: int,
    batch_index: int,
    next_batch_size: int,
    will_retry: bool,
) -> None:
    print(
        "[error] "
        f"cuda_oom=1 rank={rank} iteration={iteration_index} "
        f"batch_index={batch_index} next_batch_size={next_batch_size} "
        f"will_retry={1 if will_retry else 0}",
        file=sys.stderr,
        flush=True,
    )


def _gpu_memory_utilization(device: torch.device) -> float:
    if not torch.cuda.is_available() or device.type != "cuda":
        return 0.0
    free_bytes, total_bytes = torch.cuda.mem_get_info(device=device)
    if total_bytes <= 0:
        return 0.0
    used_ratio = 1.0 - (float(free_bytes) / float(total_bytes))
    return max(0.0, min(1.0, used_ratio))


def _init_distributed_device() -> torch.device:
    local_rank_env = os.environ.get("LOCAL_RANK")
    if local_rank_env is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    local_rank = int(local_rank_env)
    assert torch.cuda.is_available(), "LOCAL_RANK is set but CUDA is unavailable"
    assert local_rank >= 0, "LOCAL_RANK must be non-negative"
    torch.cuda.set_device(local_rank)

    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl", device_id=local_rank)
        atexit.register(_shutdown_distributed_process_group)

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
    adaptive_memory_utilization_ema: float = 0.0,
    adaptive_previous_tokens_per_sample: float = 0.0,
    adaptive_next_batch_size_float: float = 0.0,
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
        "adaptive_memory_utilization_ema": adaptive_memory_utilization_ema,
        "adaptive_previous_tokens_per_sample": adaptive_previous_tokens_per_sample,
        "adaptive_next_batch_size_float": adaptive_next_batch_size_float,
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
        _distributed_barrier()
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
    _distributed_barrier()


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
    source_model_folder = Path(source_model_path).expanduser().resolve()
    rank, _ = _get_rank_world_size()

    def _remove_existing_weight_files(model_dir: Path) -> None:
        weight_paths = [
            model_dir / "model.safetensors",
            model_dir / "model.safetensors.index.json",
            model_dir / "pytorch_model.bin",
            model_dir / "pytorch_model.bin.index.json",
        ]
        for weight_path in weight_paths:
            if weight_path.exists():
                assert weight_path.is_file(), f"weight artifact must be a file: {weight_path}"
                weight_path.unlink()

        shard_patterns = ["model-*.safetensors", "pytorch_model-*.bin"]
        for pattern in shard_patterns:
            for shard_path in model_dir.glob(pattern):
                if shard_path.exists():
                    assert shard_path.is_file(), f"weight shard must be a file: {shard_path}"
                    shard_path.unlink()

    if rank == 0:
        print(f"[status] preparing_final_output_model=1 output_parent_dir={final_model_output_parent_dir}")
        assert source_model_folder.exists(), f"source model folder does not exist: {source_model_folder}"
        assert source_model_folder.is_dir(), f"source model folder must be a directory: {source_model_folder}"
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
            shutil.rmtree(final_model_output_path)
        print(f"[status] writing_final_output_model=1 output_dir={final_model_output_path}")
        shutil.copytree(source_model_folder, final_model_output_path)
        _remove_existing_weight_files(final_model_output_path)

    _distributed_barrier()

    if training_plan == "lora_current":
        if rank == 0:
            unwrapped = model.module if hasattr(model, "module") else model
            export_model = unwrapped.merge_and_unload() if hasattr(unwrapped, "merge_and_unload") else unwrapped
            export_model.save_pretrained(
                final_model_output_path,
                safe_serialization=True,
                save_config=False,
            )
            print(f"[status] written_final_output_model=1 output_dir={final_model_output_path}")
        _distributed_barrier()
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

        export_model = AutoModelForCausalLM.from_pretrained(source_model_path, dtype=torch.bfloat16)
        incompatible = export_model.load_state_dict(state_dict, strict=True)
        assert len(incompatible.missing_keys) == 0, "final export state_dict is missing keys"
        assert len(incompatible.unexpected_keys) == 0, "final export state_dict has unexpected keys"
        export_model.save_pretrained(
            final_model_output_path,
            safe_serialization=True,
            save_config=False,
        )
        print(f"[status] written_final_output_model=1 output_dir={final_model_output_path}")
    _distributed_barrier()


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
    print(
        "[status] "
        f"rank={rank} "
        f"loading_checkpoint=1 "
        f"checkpoint_tag={checkpoint_tag} "
        f"checkpoint_dir={checkpoint_dir}"
    )
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
        adaptive_memory_utilization_ema=float(training_state.get("adaptive_memory_utilization_ema", 0.0)),
        adaptive_previous_tokens_per_sample=float(training_state.get("adaptive_previous_tokens_per_sample", 0.0)),
        adaptive_next_batch_size_float=float(
            training_state.get(
                "adaptive_next_batch_size_float",
                float(training_state.get("next_batch_size", 0)),
            )
        ),
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
    assert np.isfinite(
        resume_state.adaptive_memory_utilization_ema
    ), "resume adaptive_memory_utilization_ema must be finite"
    assert np.isfinite(
        resume_state.adaptive_previous_tokens_per_sample
    ), "resume adaptive_previous_tokens_per_sample must be finite"
    assert np.isfinite(
        resume_state.adaptive_next_batch_size_float
    ), "resume adaptive_next_batch_size_float must be finite"
    assert (
        resume_state.accumulation_step == 0
    ), "resuming from partial gradient accumulation is not supported"
    print(
        "[status] "
        f"rank={rank} "
        f"loaded_checkpoint=1 "
        f"global_step={resume_state.global_step} "
        f"next_iteration={resume_state.next_iteration_index} "
        f"next_batch_cursor={resume_state.next_batch_cursor} "
        f"next_sample_index={resume_state.next_sample_index}"
    )
    _distributed_barrier()
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
    measured_memory_utilization: float,
    measured_tokens_per_sample: float,
    target_memory_utilization: float,
    min_batch_size: int,
    max_batch_size: int,
) -> AdaptiveBatchState:
    assert measured_throughput > 0.0, "measured_throughput must be positive"
    assert measured_memory_utilization >= 0.0, "measured_memory_utilization must be non-negative"
    assert measured_tokens_per_sample > 0.0, "measured_tokens_per_sample must be positive"
    assert 0.0 < target_memory_utilization < 1.0, "target_memory_utilization must be in (0, 1)"
    assert min_batch_size > 0, "min_batch_size must be positive"
    assert max_batch_size >= min_batch_size, "max_batch_size must be >= min_batch_size"

    ema_alpha = 0.2
    momentum = 0.8
    min_improvement_ratio = 0.01
    base_step = 0.08
    max_velocity = 0.35
    memory_ema_alpha = 0.3
    max_growth_ratio = 1.2

    current_batch_size_float = adaptive_state.next_batch_size_float
    if current_batch_size_float <= 0.0:
        current_batch_size_float = float(adaptive_state.next_batch_size)
    current_batch_size_float = max(float(min_batch_size), min(float(max_batch_size), current_batch_size_float))

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

    throughput_candidate_batch_size_float = current_batch_size_float * (1.0 + velocity)
    throughput_candidate_batch_size_float = max(
        float(min_batch_size),
        min(float(max_batch_size), throughput_candidate_batch_size_float),
    )

    if throughput_candidate_batch_size_float >= float(max_batch_size) and velocity > 0.0:
        velocity = 0.0
    if throughput_candidate_batch_size_float <= float(min_batch_size) and velocity < 0.0:
        velocity = 0.0

    updated_memory_utilization_ema = adaptive_state.memory_utilization_ema
    if updated_memory_utilization_ema <= 0.0:
        updated_memory_utilization_ema = measured_memory_utilization
    else:
        updated_memory_utilization_ema = (
            (1.0 - memory_ema_alpha) * adaptive_state.memory_utilization_ema
            + memory_ema_alpha * measured_memory_utilization
        )

    safe_memory_utilization = max(updated_memory_utilization_ema, 1e-4)
    memory_scale = target_memory_utilization / safe_memory_utilization
    memory_target_batch_size_float = current_batch_size_float * memory_scale
    if int(round(memory_target_batch_size_float)) == adaptive_state.next_batch_size:
        if updated_memory_utilization_ema < target_memory_utilization:
            memory_target_batch_size_float += 1.0
        elif updated_memory_utilization_ema > target_memory_utilization:
            memory_target_batch_size_float -= 1.0

    previous_tokens_per_sample = max(adaptive_state.previous_tokens_per_sample, 1.0)
    token_growth_ratio = max(1.0, measured_tokens_per_sample / previous_tokens_per_sample)
    memory_target_batch_size_float = memory_target_batch_size_float / token_growth_ratio

    candidate_batch_size_float = (
        0.2 * throughput_candidate_batch_size_float + 0.8 * memory_target_batch_size_float
    )
    if updated_memory_utilization_ema > target_memory_utilization:
        candidate_batch_size_float = min(candidate_batch_size_float, memory_target_batch_size_float)

    max_allowed_next_float = current_batch_size_float * max_growth_ratio
    candidate_batch_size_float = min(candidate_batch_size_float, max_allowed_next_float)
    candidate_batch_size_float = max(
        float(min_batch_size),
        min(float(max_batch_size), candidate_batch_size_float),
    )

    candidate_batch_size = int(round(candidate_batch_size_float))
    candidate_batch_size = max(min_batch_size, min(max_batch_size, candidate_batch_size))

    updated_previous_tokens_per_sample = max(
        adaptive_state.previous_tokens_per_sample,
        measured_tokens_per_sample,
    )

    return AdaptiveBatchState(
        next_batch_size=candidate_batch_size,
        next_batch_size_float=candidate_batch_size_float,
        velocity=velocity,
        throughput_ema=updated_ema,
        best_throughput_ema=max(best_ema, updated_ema),
        memory_utilization_ema=updated_memory_utilization_ema,
        previous_tokens_per_sample=updated_previous_tokens_per_sample,
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


def _resolve_local_model_path(model_parent_dir: str) -> str:
    normalized_parent = Path(model_parent_dir).expanduser().resolve()
    assert normalized_parent.exists(), f"model_parent_dir does not exist: {normalized_parent}"
    assert normalized_parent.is_dir(), f"model_parent_dir must be a directory: {normalized_parent}"

    normalized = normalized_parent / "model"
    assert normalized.exists(), (
        f"model folder not found under model_parent_dir: {normalized}"
    )
    assert normalized.is_dir(), f"model folder must be a directory: {normalized}"

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
        "model_parent_dir/model must contain safetensors weights (model.safetensors or "
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
        dtype=torch.bfloat16,
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
        dtype=torch.bfloat16,
    ).to(device)
    base_model.gradient_checkpointing_enable()

    mixed_precision = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )
    return FSDP(base_model, device_id=device, mixed_precision=mixed_precision)


def train(config: TrainConfig) -> None:
    assert config.training_plan in {
        "lora_current",
        "full_fsdp_backup",
    }, "training_plan must be one of: lora_current, full_fsdp_backup"
    assert config.advantage_clip > 0.0, "advantage_clip must be positive"
    assert config.learning_rate > 0.0, "learning_rate must be positive"
    assert config.weight_decay >= 0.0, "weight_decay must be non-negative"
    assert config.num_iterations > 0, "num_iterations must be positive"
    assert config.grad_accum_steps > 0, "grad_accum_steps must be positive"
    assert config.log_time_interval > 0.0, "log_time_interval must be positive"
    assert config.checkpoint_save_time_interval > 0.0, "checkpoint_save_time_interval must be positive"
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
    target_gpu_memory_utilization = 0.90

    if _is_primary_rank():
        print(f"[status] loading_model=1 model_parent_dir={config.model_parent_dir}")
    resolved_model_path = _resolve_local_model_path(config.model_parent_dir)
    if _is_primary_rank():
        print(
            "[status] "
            f"start_training=1 training_plan={config.training_plan} "
            f"world_size={world_size} num_iterations={config.num_iterations} "
            f"model_path={resolved_model_path}"
        )
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
        adaptive_next_batch_size_float=float(initial_batch_size),
    )
    if len(resolved_resume_tag) > 0:
        if _is_primary_rank():
            print(f"[status] loading_resume_checkpoint=1 checkpoint_tag={resolved_resume_tag}")
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
                adaptive_memory_utilization_ema=resume_state.adaptive_memory_utilization_ema,
                adaptive_previous_tokens_per_sample=resume_state.adaptive_previous_tokens_per_sample,
                adaptive_next_batch_size_float=float(initial_batch_size),
            )
    elif _is_primary_rank():
        print("[status] loading_resume_checkpoint=0 starting_fresh=1")

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
                    "next_batch_size_int": initial_batch_size,
                    "next_batch_size_float": float(initial_batch_size),
                },
            )

        assert resume_state.next_batch_cursor < len(local_batches) or (
            resume_state.next_iteration_index >= config.num_iterations and resume_state.next_batch_cursor == 0
        ), "resume batch cursor is out of local batch range"

        global_step = resume_state.global_step
        accumulation_step = resume_state.accumulation_step
        last_checkpoint_save_time = time.monotonic()
        last_log_time = last_checkpoint_save_time
        optimizer.zero_grad(set_to_none=True)

        if resume_state.next_iteration_index >= config.num_iterations:
            print(
                "[status] "
                f"rank={rank} training_already_complete=1 "
                f"next_iteration={resume_state.next_iteration_index}"
            )
            _distributed_barrier()
            _shutdown_distributed_process_group()
            return

        for iteration_index in range(resume_state.next_iteration_index, config.num_iterations):
            if _is_primary_rank():
                iteration_progress = (iteration_index + 1) / config.num_iterations
                print(_json_master_progress(float(iteration_progress), f"Iteration {iteration_index + 1}/{config.num_iterations}"))
            batch_start_cursor = (
                resume_state.next_batch_cursor if iteration_index == resume_state.next_iteration_index else 0
            )
            for local_batch_cursor in range(batch_start_cursor, len(local_batches)):
                worker_progress = float(local_batch_cursor + 1) / len(local_batches)
                print(_json_worker_progress(f"rank{rank}", worker_progress, f"Batch {local_batch_cursor + 1}/{len(local_batches)}"))
                resolved_batch = local_batches[local_batch_cursor]
                should_sync = (accumulation_step + 1) == config.grad_accum_steps
                sync_context = nullcontext()
                if (
                    config.training_plan == "lora_current"
                    and world_size > 1
                    and hasattr(model, "no_sync")
                    and not should_sync
                ):
                    sync_context = model.no_sync()

                step_start = time.perf_counter()
                try:
                    collated = collate_training_samples(
                        samples=resolved_batch.samples,
                        pad_token_id=pad_token_id,
                    )

                    input_ids = collated.input_ids.to(device=device, non_blocking=True)
                    labels = collated.labels.to(device=device, non_blocking=True)
                    attention_mask = collated.attention_mask.to(device=device, non_blocking=True)
                    advantages = collated.advantages.to(device=device, non_blocking=True)

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
                except RuntimeError as exc:
                    if not _is_cuda_oom_exception(exc):
                        raise
                    optimizer.zero_grad(set_to_none=True)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    _print_cuda_oom_stderr(
                        rank=rank,
                        iteration_index=iteration_index,
                        batch_index=resolved_batch.batch_index,
                        next_batch_size=initial_batch_size,
                        will_retry=False,
                    )
                    raise

                step_elapsed_sec = max(time.perf_counter() - step_start, 1e-6)
                throughput_samples_per_sec = float(len(resolved_batch.samples)) / step_elapsed_sec
                gpu_memory_usage_pct = 100.0 * _gpu_memory_utilization(device)
                print(_json_key_value("throughput_samples_per_sec", f"{throughput_samples_per_sec:.2f}"))
                print(_json_key_value("batch_size", str(len(resolved_batch.samples))))
                print(_json_key_value("gpu_memory_usage_pct", f"{gpu_memory_usage_pct:.2f}"))
                if _is_primary_rank():
                    print(_json_key_value("global_step", str(global_step)))
                    print(_json_key_value("iteration", str(iteration_index)))
                    print(_json_key_value("batch_index", str(resolved_batch.batch_index)))
                    for stat_key, stat_value in loss_output.stats.items():
                        print(_json_key_value(stat_key, f"{stat_value:.6f}"))

                accumulation_step += 1

                if accumulation_step == config.grad_accum_steps:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    accumulation_step = 0
                    global_step += 1

                    now = time.monotonic()
                    elapsed_since_last_log_sec = now - last_log_time
                    if _is_primary_rank() and elapsed_since_last_log_sec >= config.log_time_interval:
                        log_payload: dict[str, float | int] = {
                            "step": global_step,
                            "iteration": iteration_index,
                            "batch_index": resolved_batch.batch_index,
                            "next_batch_size": initial_batch_size,
                        }
                        for key, value in loss_output.stats.items():
                            log_payload[key] = value
                        _log_json_line(logs_path, log_payload)
                        last_log_time = now

                    elapsed_since_last_checkpoint_sec = now - last_checkpoint_save_time
                    if elapsed_since_last_checkpoint_sec >= config.checkpoint_save_time_interval:
                        next_iteration_index, next_batch_cursor = _compute_next_position(
                            iteration_index=iteration_index,
                            local_batch_cursor=local_batch_cursor,
                            local_batch_count=len(local_batches),
                        )
                        checkpoint_tag = "checkpoints"
                        if _is_primary_rank():
                            print(
                                "[status] "
                                f"saving_periodic_checkpoint=1 elapsed_sec={elapsed_since_last_checkpoint_sec:.2f} "
                                f"global_step={global_step} iteration={iteration_index} "
                                f"batch_index={resolved_batch.batch_index}"
                            )
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
                        last_checkpoint_save_time = now

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

        _distributed_barrier()
        if _is_primary_rank():
            print(
                "[status] "
                f"finished_training=1 global_step={global_step} "
                f"completed_iterations={config.num_iterations} "
                f"final_model_output_parent_dir={final_model_output_parent_dir}"
            )
        _shutdown_distributed_process_group()
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
            next_batch_size_float=max(
                1.0,
                min(
                    float(lazy_loader.sample_count),
                    resume_state.adaptive_next_batch_size_float
                    if resume_state.adaptive_next_batch_size_float > 0.0
                    else float(max(1, min(lazy_loader.sample_count, resume_state.next_batch_size))),
                ),
            ),
            velocity=(
                resume_state.adaptive_velocity
                if resume_state.adaptive_velocity > 0.0
                else initial_adaptive_velocity
            ),
            throughput_ema=resume_state.adaptive_throughput_ema,
            best_throughput_ema=resume_state.adaptive_best_throughput_ema,
            memory_utilization_ema=resume_state.adaptive_memory_utilization_ema,
            previous_tokens_per_sample=resume_state.adaptive_previous_tokens_per_sample,
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
                    "next_batch_size_int": adaptive_state.next_batch_size,
                    "next_batch_size_float": adaptive_state.next_batch_size_float,
                },
            )

        global_step = resume_state.global_step
        accumulation_step = resume_state.accumulation_step
        last_checkpoint_save_time = time.monotonic()
        last_log_time = last_checkpoint_save_time
        optimizer.zero_grad(set_to_none=True)

        if resume_state.next_iteration_index >= config.num_iterations:
            print(
                "[status] "
                f"rank={rank} training_already_complete=1 "
                f"next_iteration={resume_state.next_iteration_index}"
            )
            _shutdown_distributed_process_group()
            return

        for iteration_index in range(resume_state.next_iteration_index, config.num_iterations):
            if _is_primary_rank():
                iteration_progress = (iteration_index + 1) / config.num_iterations
                print(_json_master_progress(float(iteration_progress), f"Iteration {iteration_index + 1}/{config.num_iterations}"))
            sample_index = resume_sample_index if iteration_index == resume_state.next_iteration_index else 0
            batch_index = 0
            while sample_index < lazy_loader.sample_count:
                requested_batch_size = min(adaptive_state.next_batch_size, lazy_loader.sample_count - sample_index)
                if requested_batch_size <= 0:
                    break

                worker_progress = float(sample_index) / lazy_loader.sample_count
                print(_json_worker_progress(f"rank{rank}", worker_progress, f"Sample {sample_index}/{lazy_loader.sample_count}"))

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

                should_sync = (accumulation_step + 1) == config.grad_accum_steps
                sync_context = nullcontext()

                step_start = time.perf_counter()
                try:
                    collated = collate_training_samples(
                        samples=resolved_batch.samples,
                        pad_token_id=pad_token_id,
                    )

                    input_ids = collated.input_ids.to(device=device, non_blocking=True)
                    labels = collated.labels.to(device=device, non_blocking=True)
                    attention_mask = collated.attention_mask.to(device=device, non_blocking=True)
                    advantages = collated.advantages.to(device=device, non_blocking=True)

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
                except RuntimeError as exc:
                    if not _is_cuda_oom_exception(exc):
                        raise
                    optimizer.zero_grad(set_to_none=True)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    reduced_batch_size = max(1, adaptive_state.next_batch_size // 2)
                    will_retry = adaptive_state.next_batch_size > 1
                    _print_cuda_oom_stderr(
                        rank=rank,
                        iteration_index=iteration_index,
                        batch_index=batch_index,
                        next_batch_size=reduced_batch_size,
                        will_retry=will_retry,
                    )
                    if adaptive_state.next_batch_size <= 1:
                        raise
                    adaptive_state = AdaptiveBatchState(
                        next_batch_size=reduced_batch_size,
                        next_batch_size_float=float(reduced_batch_size),
                        velocity=0.0,
                        throughput_ema=adaptive_state.throughput_ema,
                        best_throughput_ema=adaptive_state.best_throughput_ema,
                        memory_utilization_ema=adaptive_state.memory_utilization_ema,
                        previous_tokens_per_sample=adaptive_state.previous_tokens_per_sample,
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
                                "next_batch_size_float": adaptive_state.next_batch_size_float,
                            },
                        )
                    continue

                step_elapsed_sec = max(time.perf_counter() - step_start, 1e-6)
                throughput_samples_per_sec = float(len(resolved_batch.samples)) / step_elapsed_sec
                measured_tokens_per_sample = float(
                    collated.attention_mask.sum(dim=1).float().mean().item()
                )
                gpu_memory_utilization = _gpu_memory_utilization(device)
                gpu_memory_usage_pct = 100.0 * gpu_memory_utilization
                print(_json_key_value("throughput_samples_per_sec", f"{throughput_samples_per_sec:.2f}"))
                print(_json_key_value("batch_size", str(len(resolved_batch.samples))))
                print(_json_key_value("requested_batch_size", str(requested_batch_size)))
                print(_json_key_value("next_batch_size", str(adaptive_state.next_batch_size)))
                print(_json_key_value("next_batch_size_int", str(adaptive_state.next_batch_size)))
                print(_json_key_value("next_batch_size_float", f"{adaptive_state.next_batch_size_float:.2f}"))
                print(_json_key_value("gpu_memory_usage_pct", f"{gpu_memory_usage_pct:.2f}"))
                if _is_primary_rank():
                    print(_json_key_value("global_step", str(global_step)))
                    print(_json_key_value("iteration", str(iteration_index)))
                    print(_json_key_value("batch_index", str(batch_index)))
                    for stat_key, stat_value in loss_output.stats.items():
                        print(_json_key_value(stat_key, f"{stat_value:.6f}"))

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
                        measured_memory_utilization=gpu_memory_utilization,
                        measured_tokens_per_sample=measured_tokens_per_sample,
                        target_memory_utilization=target_gpu_memory_utilization,
                        min_batch_size=1,
                        max_batch_size=lazy_loader.sample_count,
                    )

                    now = time.monotonic()
                    elapsed_since_last_log_sec = now - last_log_time
                    if _is_primary_rank() and elapsed_since_last_log_sec >= config.log_time_interval:
                        log_payload: dict[str, float | int] = {
                            "step": global_step,
                            "iteration": iteration_index,
                            "batch_index": resolved_batch.batch_index,
                            "next_batch_size": adaptive_state.next_batch_size,
                            "next_batch_size_float": adaptive_state.next_batch_size_float,
                            "actual_batch_size": len(resolved_batch.samples),
                            "step_time_sec": float(step_elapsed_sec),
                            "throughput_samples_per_sec": throughput_samples_per_sec,
                            "gpu_memory_usage_pct": gpu_memory_usage_pct,
                        }
                        for key, value in loss_output.stats.items():
                            log_payload[key] = value
                        _log_json_line(logs_path, log_payload)
                        last_log_time = now

                    elapsed_since_last_checkpoint_sec = now - last_checkpoint_save_time
                    if elapsed_since_last_checkpoint_sec >= config.checkpoint_save_time_interval:
                        next_iteration_index = iteration_index
                        next_sample_index = sample_index
                        if next_sample_index >= lazy_loader.sample_count:
                            next_iteration_index += 1
                            next_sample_index = 0
                        checkpoint_tag = "checkpoints"
                        if _is_primary_rank():
                            print(
                                "[status] "
                                f"saving_periodic_checkpoint=1 elapsed_sec={elapsed_since_last_checkpoint_sec:.2f} "
                                f"global_step={global_step} iteration={iteration_index} "
                                f"batch_index={batch_index} next_sample_index={next_sample_index}"
                            )
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
                            adaptive_memory_utilization_ema=adaptive_state.memory_utilization_ema,
                            adaptive_previous_tokens_per_sample=adaptive_state.previous_tokens_per_sample,
                            adaptive_next_batch_size_float=adaptive_state.next_batch_size_float,
                        )
                        last_checkpoint_save_time = now
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
            adaptive_memory_utilization_ema=adaptive_state.memory_utilization_ema,
            adaptive_previous_tokens_per_sample=adaptive_state.previous_tokens_per_sample,
            adaptive_next_batch_size_float=adaptive_state.next_batch_size_float,
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

    _distributed_barrier()
    if _is_primary_rank():
        print(
            "[status] "
            f"finished_training=1 global_step={global_step} "
            f"completed_iterations={config.num_iterations} "
            f"final_model_output_parent_dir={final_model_output_parent_dir}"
        )
    _shutdown_distributed_process_group()
