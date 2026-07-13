from __future__ import annotations

import atexit
import copy
import gc
import json
import os
import random
import re
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from ..tui_logging import _tui_error, _tui_info, _tui_warning
from .batch_dataset import LazyResolvedBatchLoader, ResolvedTrainingBatch
from .train_loop import run_training_loop
from .training_plan import (
    DIST_STRATEGY_DDP,
    DIST_STRATEGY_FSDP,
    USE_LORA,
    USE_FULL,
    assert_supported_distributed_strategy,
    assert_supported_lora_or_full,
)


_FSDP_TRANSFORMER_BLOCK_CLASS_NAMES = {
    "BloomBlock",
    "DecoderLayer",
    "Gemma2DecoderLayer",
    "Gemma3DecoderLayer",
    "GemmaDecoderLayer",
    "GLMBlock",
    "GPTNeoXLayer",
    "GraniteDecoderLayer",
    "InternLM2DecoderLayer",
    "LlamaDecoderLayer",
    "MistralDecoderLayer",
    "MixtralDecoderLayer",
    "MPTBlock",
    "OlmoDecoderLayer",
    "Phi3DecoderLayer",
    "Qwen2DecoderLayer",
    "Qwen3DecoderLayer",
    "QwenBlock",
    "TransformerBlock",
}


@dataclass(frozen=True)
class TrainConfig:
    lora_or_full: str
    distributed_strategy: str
    model_parent_dir: str
    training_trajectory_path: str
    training_summary_parent_dir: str
    final_model_output_parent_dir: str
    advantage_clip: float
    learning_rate: float
    weight_decay: float
    training_time: float
    num_iterations_limit: int
    grad_accum_steps: int
    log_time_interval: float
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    seed: int
    adam_beta1: float
    adam_beta2: float
    lr_schedule: str
    lr_total_steps: int
    training_mode: str = "orchestration"
    oneshot_num_epochs: int = 0
    oneshot_model_output_root: str = ""


@dataclass(frozen=True)
class ResumeState:
    global_step: int
    next_iteration_index: int
    next_batch_cursor: int
    accumulation_step: int
    next_sample_index: int = 0
    elapsed_training_time_sec: float = 0.0
    samples_trained: int = 0
    samples_available: int = 0
    max_average_absolute_advantage: float = -1.0
    min_average_absolute_advantage: float = -1.0
    median_average_absolute_advantage: float = -1.0



def _reset_oneshot_epoch_resume_state(resume_state: ResumeState) -> ResumeState:
    """Reset per-epoch budget counters while preserving optimizer/data cursor state.

    One-shot multi-epoch training intentionally carries model weights, optimizer
    state, and sample cursor across epochs. However, the wall-clock budget and
    dataset-pass limit are defined per epoch, so those counters must be reset
    before starting the next epoch.
    """
    return ResumeState(
        global_step=resume_state.global_step,
        next_iteration_index=0,
        next_batch_cursor=resume_state.next_batch_cursor,
        accumulation_step=resume_state.accumulation_step,
        next_sample_index=resume_state.next_sample_index,
        elapsed_training_time_sec=0.0,
        samples_trained=resume_state.samples_trained,
        samples_available=resume_state.samples_available,
        max_average_absolute_advantage=resume_state.max_average_absolute_advantage,
        min_average_absolute_advantage=resume_state.min_average_absolute_advantage,
        median_average_absolute_advantage=resume_state.median_average_absolute_advantage,
    )


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


def _tensor_nonfinite_counts(tensor: torch.Tensor) -> tuple[int, int]:
    if not (torch.is_floating_point(tensor) or torch.is_complex(tensor)):
        return 0, 0
    nan_count = int(torch.isnan(tensor).sum().item())
    inf_count = int(torch.isinf(tensor).sum().item())
    return nan_count, inf_count


def _format_tensor_shape(tensor: torch.Tensor) -> str:
    if tensor.ndim == 0:
        return "scalar"
    return "x".join(str(int(dim)) for dim in tensor.shape)


def _iter_nested_tensors(value: Any, path: str) -> list[tuple[str, torch.Tensor]]:
    if isinstance(value, torch.Tensor):
        return [(path, value)]
    if isinstance(value, Mapping):
        tensors: list[tuple[str, torch.Tensor]] = []
        for key, nested_value in value.items():
            tensors.extend(_iter_nested_tensors(nested_value, f"{path}.{key}"))
        return tensors
    if isinstance(value, (list, tuple)):
        tensors = []
        for index, nested_value in enumerate(value):
            tensors.extend(_iter_nested_tensors(nested_value, f"{path}[{index}]"))
        return tensors
    to_tuple = getattr(value, "to_tuple", None)
    if callable(to_tuple):
        try:
            tuple_value = to_tuple()
        except Exception:
            return []
        return _iter_nested_tensors(tuple_value, path)
    return []


def _first_nonfinite_tensor_summary(value: Any, path: str) -> str | None:
    for tensor_path, tensor in _iter_nested_tensors(value, path):
        nan_count, inf_count = _tensor_nonfinite_counts(tensor)
        if nan_count == 0 and inf_count == 0:
            continue
        return (
            f"path={tensor_path} shape={_format_tensor_shape(tensor)} "
            f"dtype={tensor.dtype} nan_count={nan_count} inf_count={inf_count}"
        )
    return None


def _summarize_forward_inputs(
    input_ids: torch.Tensor, attention_mask: torch.Tensor
) -> str:
    input_ids_min = int(input_ids.min().item())
    input_ids_max = int(input_ids.max().item())
    attention_mask_min = int(attention_mask.min().item())
    attention_mask_max = int(attention_mask.max().item())
    attention_mask_nonzero = int(attention_mask.count_nonzero().item())
    return (
        f"input_ids_shape={_format_tensor_shape(input_ids)} "
        f"input_ids_dtype={input_ids.dtype} "
        f"input_ids_min={input_ids_min} input_ids_max={input_ids_max} "
        f"attention_mask_shape={_format_tensor_shape(attention_mask)} "
        f"attention_mask_dtype={attention_mask.dtype} "
        f"attention_mask_min={attention_mask_min} attention_mask_max={attention_mask_max} "
        f"attention_mask_nonzero={attention_mask_nonzero}"
    )


def trace_first_nonfinite_forward_module(
    model_engine: torch.nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor
) -> str:
    module_names = {
        module: (name if len(name) > 0 else "<root>")
        for name, module in model_engine.named_modules()
    }
    pre_hook_summaries: dict[torch.nn.Module, str | None] = {}
    first_nonfinite_output: dict[str, str] | None = None
    hook_handles: list[Any] = []

    def _pre_hook(module: torch.nn.Module, inputs: tuple[Any, ...]) -> None:
        pre_hook_summaries[module] = _first_nonfinite_tensor_summary(inputs, "input")

    def _post_hook(
        module: torch.nn.Module, inputs: tuple[Any, ...], output: Any
    ) -> None:
        nonlocal first_nonfinite_output
        if first_nonfinite_output is not None:
            return
        output_summary = _first_nonfinite_tensor_summary(output, "output")
        if output_summary is None:
            return
        first_nonfinite_output = {
            "module_name": module_names.get(module, "<unknown>"),
            "module_type": type(module).__name__,
            "input_summary": pre_hook_summaries.get(module) or "all_finite",
            "output_summary": output_summary,
        }

    for module in module_names.keys():
        hook_handles.append(module.register_forward_pre_hook(_pre_hook))
        hook_handles.append(module.register_forward_hook(_post_hook))

    try:
        with torch.no_grad():
            model_engine(
                input_ids=input_ids, attention_mask=attention_mask, use_cache=False
            )
    except Exception as exc:
        replay_error = f"forward_replay_error_type={type(exc).__name__}"
        if first_nonfinite_output is None:
            return (
                "nonfinite_forward_trace=1 status=replay_failed_before_detection "
                f"{replay_error} {_summarize_forward_inputs(input_ids, attention_mask)}"
            )
        return (
            "nonfinite_forward_trace=1 status=replay_failed_after_detection "
            f"module_name={first_nonfinite_output['module_name']} "
            f"module_type={first_nonfinite_output['module_type']} "
            f"module_input={first_nonfinite_output['input_summary']} "
            f"module_output={first_nonfinite_output['output_summary']} "
            f"{replay_error} {_summarize_forward_inputs(input_ids, attention_mask)}"
        )
    finally:
        for handle in hook_handles:
            handle.remove()

    if first_nonfinite_output is None:
        return (
            "nonfinite_forward_trace=1 status=no_nonfinite_module_found "
            f"{_summarize_forward_inputs(input_ids, attention_mask)}"
        )
    return (
        "nonfinite_forward_trace=1 status=first_nonfinite_output "
        f"module_name={first_nonfinite_output['module_name']} "
        f"module_type={first_nonfinite_output['module_type']} "
        f"module_input={first_nonfinite_output['input_summary']} "
        f"module_output={first_nonfinite_output['output_summary']} "
        f"{_summarize_forward_inputs(input_ids, attention_mask)}"
    )


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


def _resolve_attention_backend_from_env() -> str:
    raw_value = os.environ.get("TRAIN_ATTENTION_BACKEND")
    if raw_value is None:
        return "kernels-community/flash-attn2"
    normalized = raw_value.strip()
    if len(normalized) == 0:
        return "kernels-community/flash-attn2"
    normalized_lower = normalized.lower()
    assert normalized_lower in {
        "eager",
        "sdpa",
        "flash_attention_2",
        "kernels-community/flash-attn2",
    }, (
        "TRAIN_ATTENTION_BACKEND must be one of: eager, sdpa, flash_attention_2, "
        "kernels-community/flash-attn2"
    )
    return normalized_lower


def _load_causal_lm_with_attention(
    model_path: str, device: torch.device
) -> tuple[torch.nn.Module, str]:
    from transformers import AutoModelForCausalLM

    load_kwargs = {
        "dtype": torch.bfloat16,
    }
    requested_backend = _resolve_attention_backend_from_env()
    try:
        loaded_model: Any = AutoModelForCausalLM.from_pretrained(
            model_path,
            attn_implementation=requested_backend,
            **load_kwargs,
        )
    except Exception as exc:
        _release_step_memory(device)
        if _is_primary_rank():
            _tui_warning(
                "attention_backend_request_failed=1 "
                f"requested_backend={requested_backend} "
                f"error_type={type(exc).__name__}"
            )
        raise

    model = cast(torch.nn.Module, loaded_model.to(device))
    return model, requested_backend


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


_NONFINITE_TENSOR_EXCEPTION_RE = re.compile(
    r"^(?P<tensor>[a-z_]+) must be finite: nan_count=(?P<nan>\d+) inf_count=(?P<inf>\d+)(?: .*)?$"
)


def _parse_nonfinite_tensor_exception(
    exc: BaseException,
) -> tuple[str, int, int] | None:
    if not isinstance(exc, (AssertionError, RuntimeError)):
        return None
    message = str(exc).strip().lower()
    match = _NONFINITE_TENSOR_EXCEPTION_RE.match(message)
    if match is not None:
        return (
            match.group("tensor"),
            int(match.group("nan")),
            int(match.group("inf")),
        )
    if "must be finite" in message:
        return ("unknown", 0, 0)
    return None


def _is_nonfinite_logits_exception(exc: BaseException) -> bool:
    return _parse_nonfinite_tensor_exception(exc) is not None


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


def _save_final_model_folder(
    model: torch.nn.Module,
    lora_or_full: str,
    distributed_strategy: str,
    final_model_output_parent_dir: Path,
    source_model_path: str,
    tokenizer: object,
) -> None:
    lora_or_full = assert_supported_lora_or_full(lora_or_full)
    distributed_strategy = assert_supported_distributed_strategy(distributed_strategy)
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

    if distributed_strategy != DIST_STRATEGY_FSDP:
        if rank == 0:
            unwrapped = _unwrap_model(model)
            export_model: Any = unwrapped
            merge_and_unload = getattr(unwrapped, "merge_and_unload", None)
            if lora_or_full == USE_LORA and callable(merge_and_unload):
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

    assert distributed_strategy == DIST_STRATEGY_FSDP, (
        "unknown distributed strategy for final model export"
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



def _shard_batches_for_rank(
    ordered_batches: list[ResolvedTrainingBatch], rank: int, world_size: int
) -> list[ResolvedTrainingBatch]:
    assert world_size > 0, "world_size must be positive"
    assert rank >= 0, "rank must be non-negative"
    assert rank < world_size, "rank must be less than world_size"
    return [
        batch for batch in ordered_batches if batch.batch_index % world_size == rank
    ]


def _verify_tokenizer_model_match(
    *,
    model_path: str,
    tokenizer_name_or_path: str,
    ordered_batches: list[ResolvedTrainingBatch],
    model_vocab_size: int,
) -> dict[str, int | str]:
    assert len(model_path.strip()) > 0, "model_path cannot be empty"
    assert len(tokenizer_name_or_path.strip()) > 0, (
        "tokenizer_name_or_path cannot be empty"
    )
    assert model_vocab_size > 0, "model_vocab_size must be positive"
    assert len(ordered_batches) > 0, "ordered_batches cannot be empty"

    max_input_token_id = -1
    max_label_token_id = -1
    for batch in ordered_batches:
        assert batch.model_official_name == model_path, (
            "training batch model_official_name must match model_path"
        )
        for sample in batch.samples:
            for token_id in sample.input_ids:
                assert token_id >= 0, "input_ids must be non-negative"
                if token_id > max_input_token_id:
                    max_input_token_id = token_id
            for token_id in sample.labels:
                if token_id == -100:
                    continue
                assert token_id >= 0, "labels must be non-negative"
                if token_id > max_label_token_id:
                    max_label_token_id = token_id

    assert max_input_token_id < model_vocab_size, (
        "input_ids contain token id out of model vocab range"
    )
    assert max_label_token_id < model_vocab_size, (
        "labels contain token id out of model vocab range"
    )

    return {
        "model_official_name": model_path,
        "tokenizer_name_or_path": tokenizer_name_or_path,
        "model_vocab_size": model_vocab_size,
        "max_input_token_id": max_input_token_id,
        "max_label_token_id": max_label_token_id,
    }


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


LORA_TARGET_MODULES: list[str] = ["q_proj", "k_proj", "v_proj", "o_proj"]


def _build_lora_model(
    model_path: str,
    lora_rank: int,
    lora_alpha: int,
    lora_dropout: float,
    device: torch.device,
) -> tuple[torch.nn.Module, str]:
    assert lora_rank > 0, "lora_rank must be positive"
    assert lora_alpha > 0, "lora_alpha must be positive"
    assert lora_dropout >= 0.0 and lora_dropout < 1.0, "lora_dropout must be in [0, 1)"

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
        target_modules=LORA_TARGET_MODULES,
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


def _resolve_fsdp_transformer_layer_classes(
    module: torch.nn.Module,
) -> tuple[type[torch.nn.Module], ...]:
    discovered: dict[str, type[torch.nn.Module]] = {}
    for child in module.modules():
        class_name = type(child).__name__
        if class_name in _FSDP_TRANSFORMER_BLOCK_CLASS_NAMES or class_name.endswith(
            "DecoderLayer"
        ):
            discovered[class_name] = type(child)
    return tuple(discovered[class_name] for class_name in sorted(discovered.keys()))


def _wrap_as_fsdp(module: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    """Wrap a module with FSDP using the project's standard mixed-precision config.

    Callers that need an identically-sharded reference model can deepcopy an
    unwrapped module and then wrap both copies with this helper --- the identical
    sharding strategy guarantees that each GPU holds the same parameter shards
    for both models, allowing KL divergence to be computed locally per rank.
    """
    from torch.distributed.fsdp import BackwardPrefetch, MixedPrecision
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

    mixed_precision = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )
    transformer_layer_classes = _resolve_fsdp_transformer_layer_classes(module)
    auto_wrap_policy = None
    if len(transformer_layer_classes) > 0:
        auto_wrap_policy = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=set(transformer_layer_classes),
        )
        if _is_primary_rank():
            _tui_info(
                "fsdp_auto_wrap=1 "
                "policy=transformer_auto_wrap_policy "
                f"layer_classes={','.join(cls.__name__ for cls in transformer_layer_classes)}"
            )
    elif _is_primary_rank():
        _tui_warning(
            "fsdp_auto_wrap=0 policy=root_only reason=no_transformer_block_detected"
        )

    return FSDP(
        module,
        device_id=device,
        mixed_precision=mixed_precision,
        auto_wrap_policy=auto_wrap_policy,
        limit_all_gathers=True,
        forward_prefetch=False,
        backward_prefetch=BackwardPrefetch.BACKWARD_POST,
    )


def _build_fsdp_model(
    model_path: str, device: torch.device, include_raw_model: bool = False
) -> tuple[torch.nn.Module, torch.nn.Module | None, str]:
    """Build a model and return (fsdp_model, raw_model, attention_backend).

    When `include_raw_model` is true, the raw (unwrapped) model is a deepcopy of
    the base model taken *before* FSDP wrapping. FSDP mutates parameter views
    in-place (replacing them with sharded flat-parameter views), so any deepcopy
    for reference-model creation must happen first.
    """
    base_model, attention_backend = _build_full_model(
        model_path=model_path, device=device
    )
    raw_model: torch.nn.Module | None = None
    if include_raw_model:
        # Deepcopy while parameters are still plain tensors --- FSDP wrapping will
        # replace them with sharded views that are not safe to deepcopy.
        raw_model = copy.deepcopy(base_model)
    return _wrap_as_fsdp(base_model, device), raw_model, attention_backend


def _train_oneshot_multiepoch(
    *,
    config: TrainConfig,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    rank: int,
    world_size: int,
    device: torch.device,
    pad_token_id: int,
    eos_token_id: int,
    bos_token_id: int,
    model_vocab_size: int,
    expected_model_name: str,
    logs_path: Path,
    training_summary_parent_dir: str,
    resolved_model_path: str,
    tokenizer: Any,
    max_grad_norm: float,
    lr_warmup_steps: int,
    lr_min_scale: float,
    lazy_loader: LazyResolvedBatchLoader,
) -> None:
    """Multi-epoch oneshot training: all oneshot epochs in a single process.

    Model weights, optimizer state, and sample cursor persist across epochs
    entirely in memory. One-shot training does not write training-state
    checkpoints and does not support resume from prior runs.

    Each epoch runs for config.training_time seconds independently.
    """
    from . import engine as eng
    from .train_loop import _run_unified_loop

    assert config.oneshot_num_epochs > 0, "oneshot_num_epochs must be positive"

    resume_state = ResumeState(
        global_step=0,
        next_iteration_index=0,
        next_batch_cursor=0,
        accumulation_step=0,
        next_sample_index=0,
        samples_trained=0,
    )

    for oneshot_epoch in range(config.oneshot_num_epochs):
        is_final = oneshot_epoch == config.oneshot_num_epochs - 1
        epoch_number = oneshot_epoch + 1
        output_dir = (
            Path(config.oneshot_model_output_root) / f"oneshot_epoch_{epoch_number}"
        )
        epoch_start_global_step = resume_state.global_step
        epoch_start_samples_trained = resume_state.samples_trained
        epoch_start_next_sample_index = resume_state.next_sample_index

        if _is_primary_rank():
            _tui_info(
                f"oneshot_epoch={oneshot_epoch}/{config.oneshot_num_epochs} "
                f"oneshot_epoch_number={epoch_number} "
                f"output_dir={output_dir} is_final={is_final} "
                f"epoch_start_global_step={epoch_start_global_step} "
                f"epoch_start_samples_trained={epoch_start_samples_trained} "
                f"epoch_start_next_sample_index={epoch_start_next_sample_index}"
            )

        resume_state = _run_unified_loop(
            config=config,
            model=model,
            optimizer=optimizer,
            lazy_loader=lazy_loader,
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
            final_model_output_parent_dir=output_dir,
            training_summary_parent_dir=str(output_dir),
            resolved_model_path=resolved_model_path,
            tokenizer=tokenizer,
            max_grad_norm=max_grad_norm,
            lr_warmup_steps=lr_warmup_steps,
            lr_min_scale=lr_min_scale,
            eng=eng,
            finalize_training=is_final,
        )
        epoch_steps_trained = resume_state.global_step - epoch_start_global_step
        epoch_samples_trained = (
            resume_state.samples_trained - epoch_start_samples_trained
        )
        epoch_samples_available = resume_state.samples_available
        epoch_sample_coverage_pct = 0.0
        if epoch_samples_available > 0:
            epoch_sample_coverage_pct = (
                100.0 * float(epoch_samples_trained) / float(epoch_samples_available)
            )
        if _is_primary_rank():
            _tui_info(
                f"oneshot_epoch_complete=1 oneshot_epoch_number={epoch_number} "
                f"epoch_global_step_start={epoch_start_global_step} "
                f"epoch_global_step_end={resume_state.global_step} "
                f"epoch_steps_trained={epoch_steps_trained} "
                f"epoch_samples_trained={epoch_samples_trained} "
                f"epoch_samples_available={epoch_samples_available} "
                f"epoch_sample_coverage_pct={epoch_sample_coverage_pct:.4f} "
                f"epoch_elapsed_training_time_sec={resume_state.elapsed_training_time_sec:.2f} "
                f"epoch_next_sample_index_start={epoch_start_next_sample_index} "
                f"epoch_next_sample_index_end={resume_state.next_sample_index} "
                f"epoch_accumulation_step_end={resume_state.accumulation_step}"
            )
            if epoch_steps_trained <= 0 or epoch_samples_trained <= 0:
                _tui_warning(
                    f"oneshot_epoch_low_progress=1 oneshot_epoch_number={epoch_number} "
                    f"epoch_steps_trained={epoch_steps_trained} "
                    f"epoch_samples_trained={epoch_samples_trained}"
                )
        resume_state = _reset_oneshot_epoch_resume_state(resume_state)


def train(config: TrainConfig) -> None:
    lora_or_full = assert_supported_lora_or_full(config.lora_or_full)
    distributed_strategy = assert_supported_distributed_strategy(config.distributed_strategy)
    assert config.advantage_clip > 0.0, "advantage_clip must be positive"
    assert config.learning_rate > 0.0, "learning_rate must be positive"
    assert config.weight_decay >= 0.0, "weight_decay must be non-negative"
    assert config.training_time > 0.0, "training_time must be positive"
    assert config.num_iterations_limit > 0, "num_iterations_limit must be positive"
    assert config.grad_accum_steps > 0, "grad_accum_steps must be positive"
    assert config.log_time_interval > 0.0, "log_time_interval must be positive"
    assert 0.0 < config.adam_beta1 < 1.0, "adam_beta1 must be in (0, 1)"
    assert 0.0 < config.adam_beta2 < 1.0, "adam_beta2 must be in (0, 1)"
    assert len(config.training_summary_parent_dir.strip()) > 0, (
        "training_summary_parent_dir cannot be empty"
    )
    assert len(config.final_model_output_parent_dir.strip()) > 0, (
        "final_model_output_parent_dir cannot be empty"
    )

    from transformers import AutoTokenizer

    loaded_env_count = _load_dotenv_if_present()
    if loaded_env_count > 0 and _is_primary_rank():
        _tui_info(f"dotenv_loaded=1 dotenv_path=.env keys_loaded={loaded_env_count}")

    _set_seed(config.seed)
    device = _init_distributed_device()
    rank, world_size = _get_rank_world_size()
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
            f"start_training=1 lora_or_full={lora_or_full} "
            f"distributed_strategy={distributed_strategy} "
            f"world_size={world_size} training_time={config.training_time:.1f}s "
            f"model_path={resolved_model_path}"
        )
        _tui_info(
            "optimization_stability=1 "
            f"max_grad_norm={max_grad_norm:.4f} "
            f"lr_warmup_micro_batches={lr_warmup_micro_batches} "
            f"lr_warmup_steps={lr_warmup_steps} "
            f"grad_accum_steps={config.grad_accum_steps} "
            f"lr_min_scale={lr_min_scale:.4f} "
            f"lr_schedule={config.lr_schedule} "
            f"lr_total_steps={config.lr_total_steps}"
        )
        _tui_info(
            "optimizer_config=1 "
            f"adam_beta1={config.adam_beta1:.4f} "
            f"adam_beta2={config.adam_beta2:.4f} "
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

    if lora_or_full == USE_LORA:
        model, attention_backend = _build_lora_model(
            model_path=resolved_model_path,
            lora_rank=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            device=device,
        )
        raw_model = model
    else:
        assert lora_or_full == USE_FULL, f"unexpected lora_or_full value: {lora_or_full}"
        if distributed_strategy == DIST_STRATEGY_FSDP:
            model, raw_model, attention_backend = _build_fsdp_model(
                model_path=resolved_model_path,
                device=device,
            )
        else:
            model, attention_backend = _build_full_model(
                model_path=resolved_model_path, device=device
            )
            raw_model = model

    _tui_info(f"rank={rank} attention_backend={attention_backend}")

    input_embeddings = cast(Any, model).get_input_embeddings()
    assert input_embeddings is not None, "model must expose input embeddings"
    model_vocab_size = input_embeddings.num_embeddings

    # raw_model holds the unwrapped module for reference-model creation
    # (set above per training plan: model for LoRA/DDP, optional deepcopy for FSDP)

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(config.adam_beta1, config.adam_beta2),
    )

    if distributed_strategy == DIST_STRATEGY_DDP and world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device.index],
            output_device=device.index,
            find_unused_parameters=False,
        )

    training_summary_parent_dir = Path(config.training_summary_parent_dir)
    training_summary_parent_dir.mkdir(parents=True, exist_ok=True)
    final_model_output_parent_dir = Path(config.final_model_output_parent_dir)
    logs_path = training_summary_parent_dir / "train_metrics.jsonl"

    expected_model_name = resolved_model_path
    tokenizer_name = tokenizer.name_or_path.strip()
    assert len(expected_model_name) > 0, "model_path cannot be empty"
    assert len(tokenizer_name) > 0, "tokenizer_name_or_path cannot be empty"
    assert tokenizer_name == expected_model_name, (
        "tokenizer name_or_path must exactly match model_path"
    )

    resume_state = ResumeState(
        global_step=0,
        next_iteration_index=0,
        next_batch_cursor=0,
        accumulation_step=0,
        next_sample_index=0,
        samples_trained=0,
    )

    lazy_loader = LazyResolvedBatchLoader(
        training_trajectory_path=config.training_trajectory_path,
        model_official_name=expected_model_name,
        first_n_training_samples=0,
    )
    try:
        if config.training_mode == "oneshot" and config.oneshot_num_epochs > 0:
            _train_oneshot_multiepoch(
                config=config,
                model=model,
                optimizer=optimizer,
                rank=rank,
                world_size=world_size,
                device=device,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                bos_token_id=bos_token_id,
                model_vocab_size=model_vocab_size,
                expected_model_name=expected_model_name,
                logs_path=logs_path,
                training_summary_parent_dir=config.training_summary_parent_dir,
                resolved_model_path=resolved_model_path,
                tokenizer=tokenizer,
                max_grad_norm=max_grad_norm,
                lr_warmup_steps=lr_warmup_steps,
                lr_min_scale=lr_min_scale,
                lazy_loader=lazy_loader,
            )
        else:
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
                final_model_output_parent_dir=final_model_output_parent_dir,
                training_summary_parent_dir=config.training_summary_parent_dir,
                resolved_model_path=resolved_model_path,
                tokenizer=tokenizer,
                max_grad_norm=max_grad_norm,
                lr_warmup_steps=lr_warmup_steps,
                lr_min_scale=lr_min_scale,
                lazy_loader=lazy_loader,
            )
    finally:
        lazy_loader.close()
