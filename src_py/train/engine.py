from __future__ import annotations

import atexit
import gc
import json
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from ..tui_logging import _tui_error, _tui_info, _tui_warning
from .batch_dataset import LazyResolvedBatchLoader
from .train_loop import run_training_loop
from .training_plan import (
    TRAINING_PLAN_DDP,
    TRAINING_PLAN_FSDP,
    TRAINING_PLAN_LORA,
    assert_supported_training_plan,
)


@dataclass(frozen=True)
class TrainConfig:
    training_plan: str
    model_parent_dir: str
    training_trajectory_sqlite_path: str
    checkpoints_parent_dir: str
    final_model_output_parent_dir: str
    training_summary_parent_dir: str
    advantage_clip: float
    learning_rate: float
    weight_decay: float
    training_time: float
    num_iterations_limit: int
    grad_accum_steps: int
    log_time_interval: float
    checkpoint_save_time_interval: float
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    lora_target_modules_csv: str
    resume_checkpoint_tag: str
    seed: int


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
    elapsed_training_time_sec: float = 0.0
    samples_trained: int = 0
    samples_available: int = 0
    max_average_absolute_advantage: float = -1.0
    min_average_absolute_advantage: float = -1.0
    median_average_absolute_advantage: float = -1.0


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


def _load_dotenv_if_present(dotenv_path: str = ".env") -> int:
    path = Path(dotenv_path)
    if not path.exists() or not path.is_file():
        return 0

    from dotenv import dotenv_values, load_dotenv

    existing_keys = set(os.environ.keys())
    loaded = load_dotenv(dotenv_path=path, override=False)
    if not loaded:
        return 0

    parsed_values = dotenv_values(dotenv_path=path)
    loaded_count = 0
    for key in parsed_values.keys():
        if key is None:
            continue
        normalized_key = key.strip()
        if len(normalized_key) == 0:
            continue
        if normalized_key in existing_keys:
            continue
        if normalized_key in os.environ:
            loaded_count += 1
    return loaded_count


def _is_primary_rank() -> bool:
    return (
        (not torch.distributed.is_available())
        or (not torch.distributed.is_initialized())
        or torch.distributed.get_rank() == 0
    )


def _log_json_line(log_path: Path, payload: dict[str, float | int]) -> None:
    assert log_path.parent.exists(), f"log directory must exist: {log_path.parent}"
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def _forward_logits(
    model_engine: torch.nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    outputs = model_engine(
        input_ids=input_ids, attention_mask=attention_mask, use_cache=False
    )
    assert hasattr(outputs, "logits"), "model forward output must contain logits"
    logits = outputs.logits
    assert isinstance(logits, torch.Tensor), "logits must be a tensor"
    return logits


def _load_causal_lm_with_attention(
    model_path: str, device: torch.device
) -> tuple[torch.nn.Module, str]:
    from transformers import AutoModelForCausalLM

    load_kwargs = {
        "dtype": torch.bfloat16,
    }
    backend = "sdpa"
    try:
        loaded_model: Any = AutoModelForCausalLM.from_pretrained(
            model_path,
            attn_implementation=backend,
            **load_kwargs,
        )
        model = cast(torch.nn.Module, loaded_model.to(device))
        return model, backend
    except Exception as exc:
        _release_step_memory(device)
        if _is_primary_rank():
            _tui_warning(
                "attention_backend_request_failed=1 "
                f"requested_backend={backend} "
                f"error_type={type(exc).__name__}"
            )

    fallback_loaded_model: Any = AutoModelForCausalLM.from_pretrained(
        model_path,
        **load_kwargs,
    )
    model = cast(torch.nn.Module, fallback_loaded_model.to(device))
    return model, "eager"


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
    batch_token_length: int,
    next_batch_size: int,
    will_retry: bool,
    next_trajectory_length_cap: int | None = None,
) -> None:
    assert batch_token_length > 0, "batch_token_length must be positive"
    extra = ""
    if next_trajectory_length_cap is not None:
        assert next_trajectory_length_cap > 0, (
            "next_trajectory_length_cap must be positive"
        )
        extra = f" next_trajectory_length_cap={next_trajectory_length_cap}"
    _tui_error(
        f"cuda_oom=1 rank={rank} iteration={iteration_index} "
        f"batch_index={batch_index} batch_token_length={batch_token_length} "
        f"next_batch_size={next_batch_size} "
        f"will_retry={1 if will_retry else 0}{extra}"
    )


def _print_cuda_oom_diagnostics_stderr(
    *,
    rank: int,
    iteration_index: int,
    batch_index: int,
    device: torch.device,
) -> None:
    if not torch.cuda.is_available() or device.type != "cuda":
        _tui_error(
            f"cuda_oom_diagnostics=1 rank={rank} iteration={iteration_index} "
            f"batch_index={batch_index} cuda_available=0"
        )
        return

    free_bytes, total_bytes = torch.cuda.mem_get_info(device=device)
    allocated_bytes = torch.cuda.memory_allocated(device=device)
    reserved_bytes = torch.cuda.memory_reserved(device=device)
    max_allocated_bytes = torch.cuda.max_memory_allocated(device=device)
    max_reserved_bytes = torch.cuda.max_memory_reserved(device=device)

    mib = float(1024 * 1024)
    _tui_error(
        f"cuda_oom_diagnostics=1 rank={rank} iteration={iteration_index} "
        f"batch_index={batch_index} "
        f"free_mib={free_bytes / mib:.1f} total_mib={total_bytes / mib:.1f} "
        f"allocated_mib={allocated_bytes / mib:.1f} reserved_mib={reserved_bytes / mib:.1f} "
        f"max_allocated_mib={max_allocated_bytes / mib:.1f} max_reserved_mib={max_reserved_bytes / mib:.1f}"
    )


def _resolve_max_batch_size_cap_from_env() -> int | None:
    raw_value = os.environ.get("TRAIN_MAX_BATCH_SIZE")
    if raw_value is None:
        return None
    normalized = raw_value.strip()
    if len(normalized) == 0:
        return None
    parsed = int(normalized)
    assert parsed > 0, "TRAIN_MAX_BATCH_SIZE must be a positive integer when set"
    return parsed


def _resolve_reset_batch_size_on_wrap_from_env() -> bool:
    raw_value = os.environ.get("TRAIN_RESET_BATCH_SIZE_ON_WRAP")
    if raw_value is None:
        return True
    normalized = raw_value.strip().lower()
    if len(normalized) == 0:
        return True
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise AssertionError(
        "TRAIN_RESET_BATCH_SIZE_ON_WRAP must be a boolean-like value when set"
    )


def _resolve_max_grad_norm_from_env() -> float:
    raw_value = os.environ.get("TRAIN_MAX_GRAD_NORM")
    if raw_value is None:
        return 1.0
    normalized = raw_value.strip()
    if len(normalized) == 0:
        return 1.0
    parsed = float(normalized)
    assert parsed >= 0.0, "TRAIN_MAX_GRAD_NORM must be >= 0 when set"
    return parsed


def _resolve_lr_warmup_steps_from_env() -> int:
    raw_value = os.environ.get("TRAIN_LR_WARMUP_STEPS")
    if raw_value is None:
        return 100
    normalized = raw_value.strip()
    if len(normalized) == 0:
        return 100
    parsed = int(normalized)
    assert parsed >= 0, "TRAIN_LR_WARMUP_STEPS must be >= 0 when set"
    return parsed


def _resolve_lr_min_scale_from_env() -> float:
    raw_value = os.environ.get("TRAIN_LR_MIN_SCALE")
    if raw_value is None:
        return 0.1
    normalized = raw_value.strip()
    if len(normalized) == 0:
        return 0.1
    parsed = float(normalized)
    assert parsed > 0.0 and parsed <= 1.0, (
        "TRAIN_LR_MIN_SCALE must be in (0, 1] when set"
    )
    return parsed


def _release_step_memory(device: torch.device) -> None:
    gc.collect()
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.empty_cache()


def _gpu_memory_utilization(device: torch.device) -> float:
    if not torch.cuda.is_available() or device.type != "cuda":
        return 0.0
    free_bytes, total_bytes = torch.cuda.mem_get_info(device=device)
    if total_bytes <= 0:
        return 0.0
    used_ratio = 1.0 - (float(free_bytes) / float(total_bytes))
    return max(0.0, min(1.0, used_ratio))


def _gpu_memory_allocated_ratio(device: torch.device) -> float:
    if not torch.cuda.is_available() or device.type != "cuda":
        return 0.0
    total_bytes = torch.cuda.get_device_properties(device).total_memory
    if total_bytes <= 0:
        return 0.0
    allocated_bytes = torch.cuda.memory_allocated(device=device)
    used_ratio = float(allocated_bytes) / float(total_bytes)
    return max(0.0, min(1.0, used_ratio))


def _gpu_memory_peak_allocated_ratio(device: torch.device) -> float:
    if not torch.cuda.is_available() or device.type != "cuda":
        return 0.0
    total_bytes = torch.cuda.get_device_properties(device).total_memory
    if total_bytes <= 0:
        return 0.0
    peak_allocated_bytes = torch.cuda.max_memory_allocated(device=device)
    used_ratio = float(peak_allocated_bytes) / float(total_bytes)
    return max(0.0, min(1.0, used_ratio))


def _gpu_memory_reserved_ratio(device: torch.device) -> float:
    if not torch.cuda.is_available() or device.type != "cuda":
        return 0.0
    total_bytes = torch.cuda.get_device_properties(device).total_memory
    if total_bytes <= 0:
        return 0.0
    reserved_bytes = torch.cuda.memory_reserved(device=device)
    used_ratio = float(reserved_bytes) / float(total_bytes)
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


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    wrapped_module = getattr(model, "module", None)
    if isinstance(wrapped_module, torch.nn.Module):
        return wrapped_module
    return model


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
    elapsed_training_time_sec: float = 0.0,
    samples_trained: int = 0,
    samples_available: int = 0,
    max_average_absolute_advantage: float = -1.0,
    min_average_absolute_advantage: float = -1.0,
    median_average_absolute_advantage: float = -1.0,
) -> None:
    training_plan = assert_supported_training_plan(training_plan)
    assert global_step >= 0, "global_step must be non-negative"
    assert next_iteration_index >= 0, "next_iteration_index must be non-negative"
    assert next_batch_cursor >= 0, "next_batch_cursor must be non-negative"
    assert accumulation_step == 0, (
        "checkpointing with partial gradient accumulation is not supported"
    )
    assert next_sample_index >= 0, "next_sample_index must be non-negative"
    assert next_batch_size >= 0, "next_batch_size must be non-negative"
    assert elapsed_training_time_sec >= 0.0, (
        "elapsed_training_time_sec must be non-negative"
    )
    assert np.isfinite(elapsed_training_time_sec), (
        "elapsed_training_time_sec must be finite"
    )
    assert samples_trained >= 0, "samples_trained must be non-negative"
    assert samples_available >= 0, "samples_available must be non-negative"
    assert np.isfinite(max_average_absolute_advantage), (
        "max_average_absolute_advantage must be finite"
    )
    assert np.isfinite(min_average_absolute_advantage), (
        "min_average_absolute_advantage must be finite"
    )
    assert np.isfinite(median_average_absolute_advantage), (
        "median_average_absolute_advantage must be finite"
    )

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
        "elapsed_training_time_sec": elapsed_training_time_sec,
        "samples_trained": samples_trained,
        "samples_available": samples_available,
        "max_average_absolute_advantage": max_average_absolute_advantage,
        "min_average_absolute_advantage": min_average_absolute_advantage,
        "median_average_absolute_advantage": median_average_absolute_advantage,
        "accumulation_step": accumulation_step,
        "training_plan": training_plan,
        "rank": rank,
        "checkpoint_tag": checkpoint_tag,
    }
    torch.save(metadata_payload, checkpoint_dir / f"training_state.rank{rank}.pt")
    torch.save(
        optimizer.state_dict(), checkpoint_dir / f"optimizer_state.rank{rank}.pt"
    )

    if training_plan in {TRAINING_PLAN_LORA, TRAINING_PLAN_DDP}:
        if rank == 0:
            unwrapped = _unwrap_model(model)
            state_dict = (
                _extract_lora_checkpoint_state_dict(unwrapped)
                if training_plan == TRAINING_PLAN_LORA
                else unwrapped.state_dict()
            )
            torch.save(state_dict, checkpoint_dir / "model_state.pt")
            _write_latest_checkpoint_pointer(
                output_dir=output_dir, checkpoint_tag=checkpoint_tag
            )
        _distributed_barrier()
        return

    assert training_plan == TRAINING_PLAN_FSDP, (
        "unknown training plan for checkpointing"
    )
    from torch.distributed.fsdp import (
        FullStateDictConfig,
        StateDictType,
    )
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
    )

    assert isinstance(model, FSDP), "fsdp checkpoint expects FSDP model"
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
        state_dict = model.state_dict()
    if rank == 0:
        torch.save(state_dict, checkpoint_dir / "model_state.pt")
        _write_latest_checkpoint_pointer(
            output_dir=output_dir, checkpoint_tag=checkpoint_tag
        )
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
    training_plan = assert_supported_training_plan(training_plan)
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
                assert weight_path.is_file(), (
                    f"weight artifact must be a file: {weight_path}"
                )
                weight_path.unlink()

        shard_patterns = ["model-*.safetensors", "pytorch_model-*.bin"]
        for pattern in shard_patterns:
            for shard_path in model_dir.glob(pattern):
                if shard_path.exists():
                    assert shard_path.is_file(), (
                        f"weight shard must be a file: {shard_path}"
                    )
                    shard_path.unlink()

    if rank == 0:
        _tui_info(
            f"preparing_final_output_model=1 output_parent_dir={final_model_output_parent_dir}"
        )
        assert source_model_folder.exists(), (
            f"source model folder does not exist: {source_model_folder}"
        )
        assert source_model_folder.is_dir(), (
            f"source model folder must be a directory: {source_model_folder}"
        )
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
        _tui_info(f"writing_final_output_model=1 output_dir={final_model_output_path}")
        shutil.copytree(source_model_folder, final_model_output_path)
        _remove_existing_weight_files(final_model_output_path)

    _distributed_barrier()

    if training_plan in {TRAINING_PLAN_LORA, TRAINING_PLAN_DDP}:
        if rank == 0:
            unwrapped = _unwrap_model(model)
            export_model: Any = unwrapped
            merge_and_unload = getattr(unwrapped, "merge_and_unload", None)
            if training_plan == TRAINING_PLAN_LORA and callable(merge_and_unload):
                export_model = merge_and_unload()
            export_model.save_pretrained(
                final_model_output_path,
                safe_serialization=True,
                save_config=False,
            )
            _tui_info(
                f"written_final_output_model=1 output_dir={final_model_output_path}"
            )
        _distributed_barrier()
        return

    assert training_plan == TRAINING_PLAN_FSDP, (
        "unknown training plan for final model export"
    )
    from torch.distributed.fsdp import (
        FullStateDictConfig,
        StateDictType,
    )
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
    )

    assert isinstance(model, FSDP), "fsdp final export expects FSDP model"
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
        state_dict = model.state_dict()
    if rank == 0:
        from transformers import AutoModelForCausalLM

        export_model = AutoModelForCausalLM.from_pretrained(
            source_model_path, dtype=torch.bfloat16
        )
        incompatible = export_model.load_state_dict(state_dict, strict=True)
        assert len(incompatible.missing_keys) == 0, (
            "final export state_dict is missing keys"
        )
        assert len(incompatible.unexpected_keys) == 0, (
            "final export state_dict has unexpected keys"
        )
        export_model.save_pretrained(
            final_model_output_path,
            safe_serialization=True,
            save_config=False,
        )
        _tui_info(f"written_final_output_model=1 output_dir={final_model_output_path}")
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
            assert latest_path.exists(), (
                f"latest checkpoint pointer not found: {latest_path}"
            )
        return _read_latest_checkpoint_pointer(output_dir=output_dir)
    assert normalized_tag == "checkpoints", (
        "explicit resume_checkpoint_tag must be 'checkpoints' for single-epoch run layout"
    )
    return normalized_tag


def _extract_lora_checkpoint_state_dict(
    model: torch.nn.Module,
) -> dict[str, torch.Tensor]:
    if hasattr(model, "peft_config"):
        try:
            from peft import get_peft_model_state_dict

            return get_peft_model_state_dict(model)
        except ImportError:
            pass
    return model.state_dict()


def _load_lora_checkpoint_state_dict(
    model: torch.nn.Module, state_dict: dict[str, torch.Tensor]
) -> None:
    if hasattr(model, "peft_config"):
        try:
            from peft import set_peft_model_state_dict

            set_peft_model_state_dict(model, state_dict)
            return
        except ImportError:
            pass

    incompatible = model.load_state_dict(state_dict, strict=True)
    assert len(incompatible.missing_keys) == 0, "checkpoint model state is missing keys"
    assert len(incompatible.unexpected_keys) == 0, (
        "checkpoint model state has unexpected keys"
    )


def _load_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    output_dir: Path,
    checkpoint_tag: str,
    training_plan: str,
) -> ResumeState:
    training_plan = assert_supported_training_plan(training_plan)
    assert len(checkpoint_tag.strip()) > 0, "checkpoint_tag cannot be empty"

    rank, _ = _get_rank_world_size()
    checkpoint_dir = output_dir / "checkpoints"
    _tui_info(
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
    assert optimizer_state_path.exists(), (
        f"missing optimizer state: {optimizer_state_path}"
    )
    assert training_state_path.exists(), (
        f"missing training state: {training_state_path}"
    )

    model_state_dict = torch.load(model_state_path, map_location="cpu")
    incompatible: Any = None
    if training_plan in {TRAINING_PLAN_LORA, TRAINING_PLAN_DDP}:
        unwrapped = _unwrap_model(model)
        assert isinstance(model_state_dict, dict), (
            "checkpoint model state must be a state_dict"
        )
        if training_plan == TRAINING_PLAN_LORA:
            _load_lora_checkpoint_state_dict(unwrapped, model_state_dict)
        else:
            incompatible = unwrapped.load_state_dict(model_state_dict, strict=True)
    else:
        assert training_plan == TRAINING_PLAN_FSDP, (
            "unknown training plan for checkpoint loading"
        )
        from torch.distributed.fsdp import (
            FullStateDictConfig,
            StateDictType,
        )
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP,
        )

        assert isinstance(model, FSDP), "fsdp loading expects FSDP model"
        load_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=False)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, load_policy):
            incompatible = model.load_state_dict(model_state_dict, strict=True)

    if training_plan in {TRAINING_PLAN_DDP, TRAINING_PLAN_FSDP}:
        assert len(incompatible.missing_keys) == 0, (
            "checkpoint model state is missing keys"
        )
        assert len(incompatible.unexpected_keys) == 0, (
            "checkpoint model state has unexpected keys"
        )

    optimizer_state = torch.load(optimizer_state_path, map_location="cpu")
    optimizer.load_state_dict(optimizer_state)

    training_state = torch.load(training_state_path, map_location="cpu")
    checkpoint_training_plan = assert_supported_training_plan(
        str(training_state["training_plan"])
    )
    assert checkpoint_training_plan == training_plan, (
        "checkpoint training plan mismatch"
    )
    next_iteration_index_obj = training_state.get("next_iteration_index")
    if next_iteration_index_obj is None:
        next_iteration_index_obj = training_state.get("next_epoch_index")
    assert next_iteration_index_obj is not None, (
        "checkpoint missing next_iteration_index"
    )
    resume_state = ResumeState(
        global_step=int(training_state["global_step"]),
        next_iteration_index=int(next_iteration_index_obj),
        next_batch_cursor=int(training_state["next_batch_cursor"]),
        accumulation_step=int(training_state["accumulation_step"]),
        next_sample_index=int(training_state.get("next_sample_index", 0)),
        next_batch_size=int(training_state.get("next_batch_size", 0)),
        adaptive_velocity=float(training_state.get("adaptive_velocity", 0.0)),
        adaptive_throughput_ema=float(
            training_state.get("adaptive_throughput_ema", 0.0)
        ),
        adaptive_best_throughput_ema=float(
            training_state.get("adaptive_best_throughput_ema", 0.0)
        ),
        adaptive_memory_utilization_ema=float(
            training_state.get("adaptive_memory_utilization_ema", 0.0)
        ),
        adaptive_previous_tokens_per_sample=float(
            training_state.get("adaptive_previous_tokens_per_sample", 0.0)
        ),
        adaptive_next_batch_size_float=float(
            training_state.get(
                "adaptive_next_batch_size_float",
                float(training_state.get("next_batch_size", 0)),
            )
        ),
        elapsed_training_time_sec=float(
            training_state.get("elapsed_training_time_sec", 0.0)
        ),
        samples_trained=int(training_state.get("samples_trained", 0)),
        samples_available=int(training_state.get("samples_available", 0)),
        max_average_absolute_advantage=float(
            training_state.get("max_average_absolute_advantage", -1.0)
        ),
        min_average_absolute_advantage=float(
            training_state.get("min_average_absolute_advantage", -1.0)
        ),
        median_average_absolute_advantage=float(
            training_state.get("median_average_absolute_advantage", -1.0)
        ),
    )
    assert resume_state.global_step >= 0, "resume global_step must be non-negative"
    assert resume_state.next_iteration_index >= 0, (
        "resume iteration index must be non-negative"
    )
    assert resume_state.next_batch_cursor >= 0, (
        "resume batch cursor must be non-negative"
    )
    assert resume_state.next_sample_index >= 0, (
        "resume sample index must be non-negative"
    )
    assert resume_state.next_batch_size >= 0, (
        "resume next_batch_size must be non-negative"
    )
    assert np.isfinite(resume_state.adaptive_velocity), (
        "resume adaptive_velocity must be finite"
    )
    assert np.isfinite(resume_state.adaptive_throughput_ema), (
        "resume adaptive_throughput_ema must be finite"
    )
    assert np.isfinite(resume_state.adaptive_best_throughput_ema), (
        "resume adaptive_best_throughput_ema must be finite"
    )
    assert np.isfinite(resume_state.adaptive_memory_utilization_ema), (
        "resume adaptive_memory_utilization_ema must be finite"
    )
    assert np.isfinite(resume_state.adaptive_previous_tokens_per_sample), (
        "resume adaptive_previous_tokens_per_sample must be finite"
    )
    assert np.isfinite(resume_state.adaptive_next_batch_size_float), (
        "resume adaptive_next_batch_size_float must be finite"
    )
    assert np.isfinite(resume_state.elapsed_training_time_sec), (
        "resume elapsed_training_time_sec must be finite"
    )
    assert resume_state.elapsed_training_time_sec >= 0.0, (
        "resume elapsed_training_time_sec must be non-negative"
    )
    assert resume_state.samples_trained >= 0, (
        "resume samples_trained must be non-negative"
    )
    assert resume_state.samples_available >= 0, (
        "resume samples_available must be non-negative"
    )
    assert np.isfinite(resume_state.max_average_absolute_advantage), (
        "resume max_average_absolute_advantage must be finite"
    )
    assert np.isfinite(resume_state.min_average_absolute_advantage), (
        "resume min_average_absolute_advantage must be finite"
    )
    assert np.isfinite(resume_state.median_average_absolute_advantage), (
        "resume median_average_absolute_advantage must be finite"
    )
    assert resume_state.accumulation_step == 0, (
        "resuming from partial gradient accumulation is not supported"
    )
    _tui_info(
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
    assert measured_memory_utilization >= 0.0, (
        "measured_memory_utilization must be non-negative"
    )
    assert measured_tokens_per_sample > 0.0, (
        "measured_tokens_per_sample must be positive"
    )
    assert 0.0 < target_memory_utilization < 1.0, (
        "target_memory_utilization must be in (0, 1)"
    )
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
    current_batch_size_float = max(
        float(min_batch_size), min(float(max_batch_size), current_batch_size_float)
    )

    updated_ema = adaptive_state.throughput_ema
    if updated_ema <= 0.0:
        updated_ema = measured_throughput
    else:
        updated_ema = (
            1.0 - ema_alpha
        ) * adaptive_state.throughput_ema + ema_alpha * measured_throughput

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

    if (
        throughput_candidate_batch_size_float >= float(max_batch_size)
        and velocity > 0.0
    ):
        velocity = 0.0
    if (
        throughput_candidate_batch_size_float <= float(min_batch_size)
        and velocity < 0.0
    ):
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
    token_growth_ratio = max(
        1.0, measured_tokens_per_sample / previous_tokens_per_sample
    )
    memory_target_batch_size_float = memory_target_batch_size_float / token_growth_ratio

    candidate_batch_size_float = (
        0.2 * throughput_candidate_batch_size_float
        + 0.8 * memory_target_batch_size_float
    )
    if updated_memory_utilization_ema > target_memory_utilization:
        candidate_batch_size_float = min(
            candidate_batch_size_float, memory_target_batch_size_float
        )

    max_allowed_next_float = current_batch_size_float * max_growth_ratio
    candidate_batch_size_float = min(candidate_batch_size_float, max_allowed_next_float)
    candidate_batch_size_float = max(
        float(min_batch_size),
        min(float(max_batch_size), candidate_batch_size_float),
    )

    candidate_batch_size = int(round(candidate_batch_size_float))
    candidate_batch_size = max(
        min_batch_size, min(max_batch_size, candidate_batch_size)
    )

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


def _resolve_pad_token_id(
    tokenizer_pad_token_id: int | None, tokenizer_eos_token_id: int | None
) -> int:
    if tokenizer_pad_token_id is not None:
        assert tokenizer_pad_token_id >= 0, (
            "tokenizer.pad_token_id must be non-negative"
        )
        return int(tokenizer_pad_token_id)

    if tokenizer_eos_token_id is not None:
        assert tokenizer_eos_token_id >= 0, (
            "tokenizer.eos_token_id must be non-negative"
        )
        return int(tokenizer_eos_token_id)

    raise AssertionError(
        "tokenizer.pad_token_id is undefined and tokenizer.eos_token_id is also undefined; "
        "cannot resolve a padding token for training"
    )


def _normalize_optional_token_id(token_id: int | None) -> int:
    if token_id is None:
        return -1
    assert token_id >= 0, "token id must be non-negative when defined"
    return int(token_id)


def _resolve_local_model_path(model_parent_dir: str) -> str:
    normalized_parent = Path(model_parent_dir).expanduser().resolve()
    assert normalized_parent.exists(), (
        f"model_parent_dir does not exist: {normalized_parent}"
    )
    assert normalized_parent.is_dir(), (
        f"model_parent_dir must be a directory: {normalized_parent}"
    )

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


def _build_lora_model(
    model_path: str,
    lora_rank: int,
    lora_alpha: int,
    lora_dropout: float,
    lora_target_modules_csv: str,
    device: torch.device,
) -> tuple[torch.nn.Module, str]:
    assert lora_rank > 0, "lora_rank must be positive"
    assert lora_alpha > 0, "lora_alpha must be positive"
    assert lora_dropout >= 0.0 and lora_dropout < 1.0, "lora_dropout must be in [0, 1)"
    targets = [
        value.strip() for value in lora_target_modules_csv.split(",") if value.strip()
    ]
    assert len(targets) > 0, "lora_target_modules_csv must contain at least one module"

    from peft import LoraConfig, get_peft_model

    base_model, attention_backend = _load_causal_lm_with_attention(
        model_path=model_path, device=device
    )
    base_model_any: Any = base_model
    base_model_any.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=targets,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = cast(torch.nn.Module, get_peft_model(base_model_any, lora_config))
    trainable_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    assert trainable_count > 0, "LoRA model must expose trainable parameters"
    return model, attention_backend


def _build_full_model(
    model_path: str, device: torch.device
) -> tuple[torch.nn.Module, str]:
    base_model, attention_backend = _load_causal_lm_with_attention(
        model_path=model_path, device=device
    )
    base_model_any: Any = base_model
    base_model_any.gradient_checkpointing_enable()
    return base_model, attention_backend


def _build_fsdp_model(
    model_path: str, device: torch.device
) -> tuple[torch.nn.Module, str]:
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import MixedPrecision

    base_model, attention_backend = _build_full_model(
        model_path=model_path, device=device
    )

    mixed_precision = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )
    return FSDP(
        base_model, device_id=device, mixed_precision=mixed_precision
    ), attention_backend


def train(config: TrainConfig) -> None:
    training_plan = assert_supported_training_plan(config.training_plan)
    assert config.advantage_clip > 0.0, "advantage_clip must be positive"
    assert config.learning_rate > 0.0, "learning_rate must be positive"
    assert config.weight_decay >= 0.0, "weight_decay must be non-negative"
    assert config.training_time > 0.0, "training_time must be positive"
    assert config.num_iterations_limit > 0, "num_iterations_limit must be positive"
    assert config.grad_accum_steps > 0, "grad_accum_steps must be positive"
    assert config.log_time_interval > 0.0, "log_time_interval must be positive"
    assert config.checkpoint_save_time_interval > 0.0, (
        "checkpoint_save_time_interval must be positive"
    )
    assert len(config.resume_checkpoint_tag.strip()) > 0, (
        "resume_checkpoint_tag cannot be empty"
    )
    assert len(config.checkpoints_parent_dir.strip()) > 0, (
        "checkpoints_parent_dir cannot be empty"
    )
    assert len(config.final_model_output_parent_dir.strip()) > 0, (
        "final_model_output_parent_dir cannot be empty"
    )
    assert len(config.training_summary_parent_dir.strip()) > 0, (
        "training_summary_parent_dir cannot be empty"
    )

    from transformers import AutoTokenizer

    loaded_env_count = _load_dotenv_if_present()
    if loaded_env_count > 0 and _is_primary_rank():
        _tui_info(f"dotenv_loaded=1 dotenv_path=.env keys_loaded={loaded_env_count}")

    _set_seed(config.seed)
    device = _init_distributed_device()
    rank, world_size = _get_rank_world_size()
    initial_batch_size = 1
    initial_adaptive_velocity = 0.12
    target_gpu_memory_utilization = 0.8
    max_batch_size_cap = _resolve_max_batch_size_cap_from_env()
    reset_batch_size_on_wrap = _resolve_reset_batch_size_on_wrap_from_env()
    max_grad_norm = _resolve_max_grad_norm_from_env()
    lr_warmup_micro_batches = _resolve_lr_warmup_steps_from_env()
    lr_warmup_steps = 0
    if lr_warmup_micro_batches > 0:
        lr_warmup_steps = max(1, lr_warmup_micro_batches // config.grad_accum_steps)
    lr_min_scale = _resolve_lr_min_scale_from_env()

    if _is_primary_rank():
        _tui_info(f"loading_model=1 model_parent_dir={config.model_parent_dir}")
    resolved_model_path = _resolve_local_model_path(config.model_parent_dir)
    if _is_primary_rank():
        _tui_info(
            f"start_training=1 training_plan={training_plan} "
            f"world_size={world_size} training_time={config.training_time:.1f}s "
            f"model_path={resolved_model_path}"
        )
        _tui_info(
            "adaptive_batch_cap=1 "
            f"train_max_batch_size_env={max_batch_size_cap if max_batch_size_cap is not None else 'unset'}"
        )
        _tui_info(
            "adaptive_batch_wrap_reset=1 "
            f"train_reset_batch_size_on_wrap={1 if reset_batch_size_on_wrap else 0}"
        )
        _tui_info(
            "optimization_stability=1 "
            f"max_grad_norm={max_grad_norm:.4f} "
            f"lr_warmup_micro_batches={lr_warmup_micro_batches} "
            f"lr_warmup_steps={lr_warmup_steps} "
            f"grad_accum_steps={config.grad_accum_steps} "
            f"lr_min_scale={lr_min_scale:.4f}"
        )
    tokenizer = AutoTokenizer.from_pretrained(resolved_model_path)
    eos_token_id = _normalize_optional_token_id(tokenizer.eos_token_id)
    pad_token_id = _resolve_pad_token_id(tokenizer.pad_token_id, tokenizer.eos_token_id)
    bos_token_id = _normalize_optional_token_id(tokenizer.bos_token_id)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = int(tokenizer.eos_token_id)
        if _is_primary_rank():
            _tui_info(
                "tokenizer_pad_token_fallback=1 "
                f"fallback_source=eos_token_id pad_token_id={tokenizer.pad_token_id}"
            )

    if training_plan == TRAINING_PLAN_LORA:
        model, attention_backend = _build_lora_model(
            model_path=resolved_model_path,
            lora_rank=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            lora_target_modules_csv=config.lora_target_modules_csv,
            device=device,
        )
    elif training_plan == TRAINING_PLAN_DDP:
        model, attention_backend = _build_full_model(
            model_path=resolved_model_path, device=device
        )
    else:
        model, attention_backend = _build_fsdp_model(
            model_path=resolved_model_path, device=device
        )

    _tui_info(f"rank={rank} attention_backend={attention_backend}")

    input_embeddings = cast(Any, model).get_input_embeddings()
    assert input_embeddings is not None, "model must expose input embeddings"
    model_vocab_size = input_embeddings.num_embeddings

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.95),
    )

    if training_plan in {TRAINING_PLAN_LORA, TRAINING_PLAN_DDP} and world_size > 1:
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
    assert tokenizer_name == expected_model_name, (
        "tokenizer name_or_path must exactly match model_path"
    )

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
        samples_trained=0,
    )
    if len(resolved_resume_tag) > 0:
        if _is_primary_rank():
            _tui_info(
                f"loading_resume_checkpoint=1 checkpoint_tag={resolved_resume_tag}"
            )
        resume_state = _load_checkpoint(
            model=model,
            optimizer=optimizer,
            output_dir=checkpoints_parent_dir,
            checkpoint_tag=resolved_resume_tag,
            training_plan=training_plan,
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
                elapsed_training_time_sec=resume_state.elapsed_training_time_sec,
                samples_trained=resume_state.samples_trained,
                samples_available=resume_state.samples_available,
                max_average_absolute_advantage=resume_state.max_average_absolute_advantage,
                min_average_absolute_advantage=resume_state.min_average_absolute_advantage,
                median_average_absolute_advantage=resume_state.median_average_absolute_advantage,
            )
    elif _is_primary_rank():
        _tui_info("loading_resume_checkpoint=0 starting_fresh=1")

    lazy_loader = LazyResolvedBatchLoader(
        training_trajectory_sqlite_path=config.training_trajectory_sqlite_path,
        model_official_name=expected_model_name,
        first_n_training_samples=0,
    )
    try:
        run_training_loop(
            config=config,
            model=model,
            optimizer=optimizer,
            resume_state=resume_state,
            rank=rank,
            world_size=world_size,
            device=device,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            bos_token_id=bos_token_id,
            model_vocab_size=model_vocab_size,
            expected_model_name=expected_model_name,
            logs_path=logs_path,
            checkpoints_parent_dir=checkpoints_parent_dir,
            final_model_output_parent_dir=final_model_output_parent_dir,
            training_summary_parent_dir=config.training_summary_parent_dir,
            resolved_model_path=resolved_model_path,
            tokenizer=tokenizer,
            initial_batch_size=initial_batch_size,
            initial_adaptive_velocity=initial_adaptive_velocity,
            target_gpu_memory_utilization=target_gpu_memory_utilization,
            max_batch_size_cap=max_batch_size_cap,
            reset_batch_size_on_wrap=reset_batch_size_on_wrap,
            max_grad_norm=max_grad_norm,
            lr_warmup_steps=lr_warmup_steps,
            lr_min_scale=lr_min_scale,
            lazy_loader=lazy_loader,
        )
    finally:
        lazy_loader.close()
