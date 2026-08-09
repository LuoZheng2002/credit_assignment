from __future__ import annotations

import json
import math
import statistics
import sys
import time
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch

from ..text_logging import (
    _text_delete_worker_bar,
    _text_error,
    _text_info,
    _text_key_value,
    _text_master_progress,
    _text_warning,
    _text_worker_progress,
)
from .batch_dataset import LazyResolvedBatchLoader, ResolvedTrainingBatch
from .collator import IGNORE_LABEL, collate_training_samples
from .data_msgpack import QuestionNodeId, TrainingSampleTokenized
from .losses import compute_advantage_weighted_causal_lm_loss
from .training_plan import (
    DIST_STRATEGY_DDP,
    DIST_STRATEGY_FSDP,
    DIST_STRATEGY_SINGLE_GPU,
    assert_supported_distributed_strategy,
    assert_supported_lora_or_full,
)


@dataclass
class TrainingLoopClock:
    training_time: float
    resumed_elapsed_training_time_sec: float
    run_start_time: float
    last_checkpoint_save_time: float
    last_log_time: float
    last_master_progress_time: float


@dataclass(frozen=True)
class DistributedStepControl:
    opcode: int
    requested_batch_size: int
    global_sample_cursor: int
    iteration_index: int


@dataclass(frozen=True)
class CudaOomPlan:
    stop_training: bool
    next_iteration_index: int



DEFAULT_TRAJECTORY_LENGTH_CAP = 4096
MIN_TRAJECTORY_LENGTH_CAP = 2
SINGLE_GPU_LORA_SYNTHETIC_OOM_PRECHECK_SAFETY_MULTIPLIER = 0.75
_DISTRIBUTED_CONTROL_STOP = -1
_DISTRIBUTED_CONTROL_SKIP = 0
_DISTRIBUTED_CONTROL_RUN = 1


def _memtrace(message: str) -> None:
    print(f"[memtrace] {message}", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()


def _truncate_sample_to_cap(
    sample: TrainingSampleTokenized, trajectory_length_cap: int
) -> TrainingSampleTokenized:
    assert trajectory_length_cap >= MIN_TRAJECTORY_LENGTH_CAP, (
        "trajectory_length_cap too small"
    )
    if sample.input_length <= trajectory_length_cap:
        return sample

    start = sample.input_length - trajectory_length_cap
    truncated_input_ids = sample.input_ids[start:]
    truncated_labels = sample.labels[start:]
    assert len(truncated_input_ids) == len(truncated_labels), (
        "truncated token arrays must align"
    )
    assert len(truncated_input_ids) >= MIN_TRAJECTORY_LENGTH_CAP, (
        "truncated sample is too short"
    )
    truncated_token_advantages = sample.token_advantages[start:]
    assert len(truncated_input_ids) == len(truncated_token_advantages), (
        "truncated token advantages must align"
    )
    truncated_old_logprobs = sample.old_logprobs[start:]
    assert len(truncated_input_ids) == len(truncated_old_logprobs), (
        "truncated old logprobs must align"
    )
    truncated_ref_logprobs = (
        sample.ref_logprobs[start:] if sample.ref_logprobs is not None else None
    )
    if truncated_ref_logprobs is not None:
        assert len(truncated_input_ids) == len(truncated_ref_logprobs), (
            "truncated ref logprobs must align"
        )
    return TrainingSampleTokenized(
        id=sample.id,
        input_ids=truncated_input_ids,
        labels=truncated_labels,
        input_length=len(truncated_input_ids),
        token_advantages=truncated_token_advantages,
        old_logprobs=truncated_old_logprobs,
        ref_logprobs=truncated_ref_logprobs,
        model_official_name=sample.model_official_name,
    )


def _truncate_samples_to_cap(
    samples: list[TrainingSampleTokenized], trajectory_length_cap: int
) -> list[TrainingSampleTokenized]:
    assert len(samples) > 0, "samples cannot be empty"
    return [
        _truncate_sample_to_cap(sample, trajectory_length_cap) for sample in samples
    ]


def _emit_trajectory_length_cap(*, cap: int) -> None:
    _text_key_value("trajectory_length_cap", str(cap))


def _tensor_diagnostic_fragment(name: str, tensor: torch.Tensor | None) -> str:
    if tensor is None:
        return f"{name}=None"

    shape = (
        "scalar"
        if tensor.ndim == 0
        else "x".join(str(int(dim)) for dim in tensor.shape)
    )
    return f"{name}_shape={shape} {name}_dtype={tensor.dtype}"


_NONFINITE_TENSOR_CODES = {
    "logits": 1,
    "advantages": 2,
    "loss": 3,
    "token_losses": 3,
    "weighted_loss": 3,
    "total_loss": 3,
    "gradients": 4,
    "grad_norm": 4,
    "optimizer_state": 5,
}


def _nonfinite_tensor_code(tensor_name: str) -> int:
    return _NONFINITE_TENSOR_CODES.get(tensor_name, 0)


def _is_primary_rank() -> bool:
    return (
        (not torch.distributed.is_available())
        or (not torch.distributed.is_initialized())
        or torch.distributed.get_rank() == 0
    )


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
    if isinstance(value, dict):
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


def _summarize_backward_inputs(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    advantages: torch.Tensor,
) -> str:
    return (
        f"input_ids_shape={_format_tensor_shape(input_ids)} "
        f"attention_mask_shape={_format_tensor_shape(attention_mask)} "
        f"labels_shape={_format_tensor_shape(labels)} "
        f"advantages_shape={_format_tensor_shape(advantages)}"
    )


def trace_first_nonfinite_backward_signal(
    *,
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    advantages: torch.Tensor,
    old_logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor | None,
    advantage_clip: float,
    kl_beta: float,
    min_grad_accum_steps: int,
) -> str:
    module_names = {
        module: (name if len(name) > 0 else "<root>")
        for name, module in model.named_modules()
    }
    first_nonfinite_module: dict[str, str] | None = None
    first_nonfinite_parameter: dict[str, str] | None = None
    hook_handles: list[Any] = []

    def _module_backward_hook(
        module: torch.nn.Module,
        grad_input: Any,
        grad_output: Any,
    ) -> Any:
        nonlocal first_nonfinite_module
        if first_nonfinite_module is not None:
            return
        grad_input_summary = _first_nonfinite_tensor_summary(grad_input, "grad_input")
        grad_output_summary = _first_nonfinite_tensor_summary(
            grad_output, "grad_output"
        )
        if grad_input_summary is None and grad_output_summary is None:
            return
        first_nonfinite_module = {
            "module_name": module_names.get(module, "<unknown>"),
            "module_type": type(module).__name__,
            "grad_input": grad_input_summary or "all_finite",
            "grad_output": grad_output_summary or "all_finite",
        }

    def _make_parameter_hook(parameter_name: str):
        def _parameter_hook(gradient: torch.Tensor) -> torch.Tensor:
            nonlocal first_nonfinite_parameter
            if first_nonfinite_parameter is None:
                gradient_summary = _first_nonfinite_tensor_summary(gradient, "grad")
                if gradient_summary is not None:
                    first_nonfinite_parameter = {
                        "parameter_name": parameter_name,
                        "gradient": gradient_summary,
                    }
            return gradient

        return _parameter_hook

    for module in module_names.keys():
        hook_handles.append(module.register_full_backward_hook(_module_backward_hook))
    for parameter_name, parameter in model.named_parameters():
        if parameter.requires_grad:
            hook_handles.append(
                parameter.register_hook(_make_parameter_hook(parameter_name))
            )

    try:
        outputs = model(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=False
        )
        logits: Any = None
        if hasattr(outputs, "logits"):
            logits = outputs.logits
        elif isinstance(outputs, dict) and "logits" in outputs:
            logits = outputs["logits"]
        assert isinstance(logits, torch.Tensor), (
            "model forward output must contain tensor logits"
        )
        loss_output = compute_advantage_weighted_causal_lm_loss(
            logits=logits,
            labels=labels,
            advantages=advantages,
            old_logprobs=old_logprobs,
            ref_logprobs=ref_logprobs,
            advantage_clip=advantage_clip,
            kl_beta=kl_beta,
        )
        replay_loss = loss_output.loss / min_grad_accum_steps
        replay_loss.backward()
    except Exception as exc:
        detail_parts = [
            "nonfinite_backward_trace=1",
            "status=replay_failed",
            f"error_type={type(exc).__name__}",
        ]
        if first_nonfinite_module is not None:
            detail_parts.extend(
                [
                    f"module_name={first_nonfinite_module['module_name']}",
                    f"module_type={first_nonfinite_module['module_type']}",
                    f"module_grad_input={first_nonfinite_module['grad_input']}",
                    f"module_grad_output={first_nonfinite_module['grad_output']}",
                ]
            )
        if first_nonfinite_parameter is not None:
            detail_parts.extend(
                [
                    f"parameter_name={first_nonfinite_parameter['parameter_name']}",
                    f"parameter_grad={first_nonfinite_parameter['gradient']}",
                ]
            )
        detail_parts.append(
            _summarize_backward_inputs(input_ids, attention_mask, labels, advantages)
        )
        return " ".join(detail_parts)
    finally:
        for handle in hook_handles:
            handle.remove()

    detail_parts = ["nonfinite_backward_trace=1"]
    if first_nonfinite_module is None and first_nonfinite_parameter is None:
        detail_parts.extend(
            [
                "status=no_nonfinite_backward_signal_found",
                _summarize_backward_inputs(
                    input_ids, attention_mask, labels, advantages
                ),
            ]
        )
        return " ".join(detail_parts)

    detail_parts.append("status=first_nonfinite_backward_signal")
    if first_nonfinite_module is not None:
        detail_parts.extend(
            [
                f"module_name={first_nonfinite_module['module_name']}",
                f"module_type={first_nonfinite_module['module_type']}",
                f"module_grad_input={first_nonfinite_module['grad_input']}",
                f"module_grad_output={first_nonfinite_module['grad_output']}",
            ]
        )
    if first_nonfinite_parameter is not None:
        detail_parts.extend(
            [
                f"parameter_name={first_nonfinite_parameter['parameter_name']}",
                f"parameter_grad={first_nonfinite_parameter['gradient']}",
            ]
        )
    detail_parts.append(
        _summarize_backward_inputs(input_ids, attention_mask, labels, advantages)
    )
    return " ".join(detail_parts)


def _assert_gradient_tensors_finite(model: torch.nn.Module) -> None:
    total_nan_count = 0
    total_inf_count = 0
    first_nonfinite_detail: str | None = None
    for parameter_name, parameter in model.named_parameters():
        gradient = parameter.grad
        if gradient is None:
            continue
        nan_count, inf_count = _tensor_nonfinite_counts(gradient)
        total_nan_count += nan_count
        total_inf_count += inf_count
        if first_nonfinite_detail is None and (nan_count > 0 or inf_count > 0):
            shape = (
                "scalar"
                if gradient.ndim == 0
                else "x".join(str(int(dim)) for dim in gradient.shape)
            )
            first_nonfinite_detail = (
                f"first_nonfinite=parameter:{parameter_name} "
                f"shape={shape} dtype={gradient.dtype}"
            )
    assert total_nan_count == 0 and total_inf_count == 0, (
        f"gradients must be finite: nan_count={total_nan_count} inf_count={total_inf_count}"
        + (f" {first_nonfinite_detail}" if first_nonfinite_detail is not None else "")
    )


def _assert_scalar_finite(value: float, tensor_name: str, detail: str) -> None:
    nan_count = 1 if math.isnan(value) else 0
    inf_count = 1 if math.isinf(value) else 0
    assert nan_count == 0 and inf_count == 0, (
        f"{tensor_name} must be finite: nan_count={nan_count} inf_count={inf_count} {detail}"
    )


def _assert_optimizer_state_finite(
    model: torch.nn.Module, optimizer: torch.optim.Optimizer
) -> None:
    parameter_names = {
        id(parameter): parameter_name
        for parameter_name, parameter in model.named_parameters()
    }
    total_nan_count = 0
    total_inf_count = 0
    first_nonfinite_detail: str | None = None
    for parameter, state in optimizer.state.items():
        parameter_name = parameter_names.get(id(parameter), "<unknown>")
        assert isinstance(state, dict), "optimizer state entries must be dictionaries"
        for state_key, state_value in state.items():
            if isinstance(state_value, torch.Tensor):
                nan_count, inf_count = _tensor_nonfinite_counts(state_value)
                total_nan_count += nan_count
                total_inf_count += inf_count
                if first_nonfinite_detail is None and (nan_count > 0 or inf_count > 0):
                    shape = (
                        "scalar"
                        if state_value.ndim == 0
                        else "x".join(str(int(dim)) for dim in state_value.shape)
                    )
                    first_nonfinite_detail = (
                        f"first_nonfinite=parameter:{parameter_name} "
                        f"state_key={state_key} shape={shape} dtype={state_value.dtype}"
                    )
            elif isinstance(state_value, (float, int)):
                numeric_value = float(state_value)
                if not math.isfinite(numeric_value):
                    total_nan_count += 1 if math.isnan(numeric_value) else 0
                    total_inf_count += 1 if math.isinf(numeric_value) else 0
                    if first_nonfinite_detail is None:
                        first_nonfinite_detail = (
                            f"first_nonfinite=parameter:{parameter_name} state_key={state_key} "
                            f"value={numeric_value}"
                        )
    assert total_nan_count == 0 and total_inf_count == 0, (
        f"optimizer_state must be finite: nan_count={total_nan_count} inf_count={total_inf_count}"
        + (f" {first_nonfinite_detail}" if first_nonfinite_detail is not None else "")
    )


def _assert_pre_step_finite(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    clipped_grad_norm: float | None,
) -> None:
    if clipped_grad_norm is not None:
        _assert_scalar_finite(
            clipped_grad_norm,
            "grad_norm",
            f"value={clipped_grad_norm}",
        )


def _extract_nonfinite_trace_suffix(message: str) -> str:
    for marker in [" nonfinite_backward_trace=1", " nonfinite_forward_trace=1"]:
        marker_index = message.find(marker)
        if marker_index >= 0:
            return message[marker_index:]
    return ""


def _init_training_loop_clock(
    *, training_time: float, resumed_elapsed_training_time_sec: float
) -> TrainingLoopClock:
    run_start_time = time.monotonic()
    return TrainingLoopClock(
        training_time=training_time,
        resumed_elapsed_training_time_sec=resumed_elapsed_training_time_sec,
        run_start_time=run_start_time,
        last_checkpoint_save_time=run_start_time,
        last_log_time=run_start_time,
        last_master_progress_time=run_start_time - 1.0,
    )


def _should_continue_training_iterations(
    *, iteration_index: int, num_iterations_limit: int
) -> bool:
    return iteration_index < num_iterations_limit


def _elapsed_training_time_sec(
    *, clock: TrainingLoopClock, now: float | None = None
) -> float:
    if now is None:
        now = time.monotonic()
    return clock.resumed_elapsed_training_time_sec + (now - clock.run_start_time)


def _plan_cuda_oom(*, iteration_index: int) -> CudaOomPlan:
    next_iteration_index = iteration_index + 1
    return CudaOomPlan(
        stop_training=True,
        next_iteration_index=next_iteration_index,
    )


def _should_fail_fast_on_cuda_oom(
    *, distributed_strategy: str, world_size: int, training_set_sort_mode: str
) -> bool:
    if training_set_sort_mode == "ByQuestion":
        return True
    normalized_strategy = assert_supported_distributed_strategy(distributed_strategy)
    if world_size <= 1:
        return False
    return normalized_strategy in {DIST_STRATEGY_DDP, DIST_STRATEGY_FSDP}


def _synthetic_probe_token_id(
    *, bos_token_id: int, eos_token_id: int, pad_token_id: int, model_vocab_size: int
) -> int:
    for token_id in (bos_token_id, eos_token_id, pad_token_id, 0):
        if 0 <= token_id < model_vocab_size:
            return token_id
    raise AssertionError("no valid synthetic token id available for OOM preflight")


def _make_synthetic_zero_advantage_sample(
    *,
    sequence_length: int,
    token_id: int,
    model_official_name: str,
) -> TrainingSampleTokenized:
    assert sequence_length >= MIN_TRAJECTORY_LENGTH_CAP, "sequence_length too small"
    input_ids = [token_id] * sequence_length
    labels = input_ids.copy()
    labels[0] = IGNORE_LABEL
    return TrainingSampleTokenized(
        id=QuestionNodeId(question_id=-1, node_id=-1),
        input_ids=input_ids,
        labels=labels,
        input_length=sequence_length,
        token_advantages=[0.0] * sequence_length,
        old_logprobs=[0.0] * sequence_length,
        ref_logprobs=[0.0] * sequence_length,
        model_official_name=model_official_name,
    )


def _probe_synthetic_sequence_length(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    sequence_length: int,
    token_id: int,
    device: torch.device,
    pad_token_id: int,
    expected_model_name: str,
    advantage_clip: float,
    min_grad_accum_steps: int,
    eng: Any,
) -> bool:
    sample = _make_synthetic_zero_advantage_sample(
        sequence_length=sequence_length,
        token_id=token_id,
        model_official_name=expected_model_name,
    )
    collated = input_ids = labels = attention_mask = advantages = old_logprobs = ref_logprobs = logits = loss_output = loss = None
    try:
        collated = collate_training_samples(samples=[sample], pad_token_id=pad_token_id)
        input_ids = collated.input_ids.to(device=device, non_blocking=True)
        labels = collated.labels.to(device=device, non_blocking=True)
        attention_mask = collated.attention_mask.to(device=device, non_blocking=True)
        advantages = collated.advantages.to(device=device, non_blocking=True)
        old_logprobs = collated.old_logprobs.to(device=device, non_blocking=True)
        ref_logprobs = (
            collated.ref_logprobs.to(device=device, non_blocking=True)
            if collated.ref_logprobs is not None
            else None
        )
        logits = eng._forward_logits(
            model, input_ids=input_ids, attention_mask=attention_mask
        )
        loss_output = compute_advantage_weighted_causal_lm_loss(
            logits=logits,
            labels=labels,
            advantages=advantages,
            old_logprobs=old_logprobs,
            ref_logprobs=ref_logprobs,
            advantage_clip=advantage_clip,
            kl_beta=0.0,
        )
        loss = loss_output.loss / min_grad_accum_steps
        loss.backward()
        optimizer.zero_grad(set_to_none=True)
        return True
    except (RuntimeError, AssertionError) as exc:
        if eng._is_cuda_oom_exception(exc):
            optimizer.zero_grad(set_to_none=True)
            return False
        raise
    finally:
        collated = None
        input_ids = None
        labels = None
        attention_mask = None
        advantages = None
        old_logprobs = None
        ref_logprobs = None
        logits = None
        loss_output = None
        loss = None
        eng._release_step_memory(device)


def _maybe_apply_single_gpu_lora_synthetic_oom_preflight(
    *,
    config: Any,
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
    resume_state: Any,
    eng: Any,
) -> int:
    trajectory_length_cap = min(
        DEFAULT_TRAJECTORY_LENGTH_CAP, config.training_trajectory_len_cutoff
    )
    lora_or_full = assert_supported_lora_or_full(config.lora_or_full)
    distributed_strategy = assert_supported_distributed_strategy(config.distributed_strategy)
    global_step = int(getattr(resume_state, "global_step", 0))
    samples_trained = int(getattr(resume_state, "samples_trained", 0))
    elapsed_training_time_sec = float(
        getattr(resume_state, "elapsed_training_time_sec", 0.0)
    )
    samples_trained_this_run = int(getattr(resume_state, "samples_trained_this_run", 0))
    should_run_preflight = (
        rank == 0
        and world_size == 1
        and distributed_strategy == DIST_STRATEGY_SINGLE_GPU
        and lora_or_full == "lora"
        and trajectory_length_cap > MIN_TRAJECTORY_LENGTH_CAP
        and elapsed_training_time_sec <= 0.0
        and samples_trained_this_run == 0
    )
    if not should_run_preflight:
        if rank == 0:
            _memtrace(
                "synthetic_oom_preflight_skipped "
                f"rank={rank} world_size={world_size} "
                f"distributed_strategy={distributed_strategy} "
                f"lora_or_full={lora_or_full} "
                f"trajectory_length_cap={trajectory_length_cap} "
                f"global_step={global_step} samples_trained={samples_trained} "
                f"elapsed_training_time_sec={elapsed_training_time_sec} "
                f"samples_trained_this_run={samples_trained_this_run}"
            )
        return trajectory_length_cap

    token_id = _synthetic_probe_token_id(
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        model_vocab_size=model_vocab_size,
    )
    low = MIN_TRAJECTORY_LENGTH_CAP
    high = trajectory_length_cap
    precision = 16
    _memtrace(
        "synthetic_oom_preflight_start "
        f"high={high} low={low} precision={precision} "
        "enabled_for=single_gpu_lora"
    )
    _text_info(
        "synthetic_oom_preflight_start=1 "
        f"high={high} low={low} precision={precision} "
        "enabled_for=single_gpu_lora"
    )
    if _probe_synthetic_sequence_length(
        model=model,
        optimizer=optimizer,
        sequence_length=high,
        token_id=token_id,
        device=device,
        pad_token_id=pad_token_id,
        expected_model_name=expected_model_name,
        advantage_clip=config.advantage_clip,
        min_grad_accum_steps=config.min_grad_accum_steps,
        eng=eng,
    ):
        longest_valid = high
    else:
        longest_valid = low
        while high - low > precision:
            mid = (low + high) // 2
            ok = _probe_synthetic_sequence_length(
                model=model,
                optimizer=optimizer,
                sequence_length=mid,
                token_id=token_id,
                device=device,
                pad_token_id=pad_token_id,
                expected_model_name=expected_model_name,
                advantage_clip=config.advantage_clip,
                min_grad_accum_steps=config.min_grad_accum_steps,
                eng=eng,
            )
            _text_info(
                "synthetic_oom_preflight_probe=1 "
                f"sequence_length={mid} ok={int(ok)} low={low} high={high}"
            )
            _memtrace(
                "synthetic_oom_preflight_probe "
                f"sequence_length={mid} ok={int(ok)} low={low} high={high}"
            )
            if ok:
                low = mid
                longest_valid = mid
            else:
                high = mid
    safety_multiplier = SINGLE_GPU_LORA_SYNTHETIC_OOM_PRECHECK_SAFETY_MULTIPLIER
    selected_cap = max(
        MIN_TRAJECTORY_LENGTH_CAP,
        min(trajectory_length_cap, int(longest_valid * safety_multiplier)),
    )
    _text_info(
        "synthetic_oom_preflight_result=1 "
        f"longest_valid={longest_valid} precision={precision} "
        f"safety_multiplier={safety_multiplier} "
        f"selected_training_trajectory_len_cutoff={selected_cap}"
    )
    _memtrace(
        "synthetic_oom_preflight_result "
        f"longest_valid={longest_valid} precision={precision} "
        f"safety_multiplier={safety_multiplier} "
        f"selected_training_trajectory_len_cutoff={selected_cap}"
    )
    return selected_cap


def _plan_distributed_step_control(
    *,
    num_iterations_limit: int,
    iteration_index: int,
    sample_count: int,
    global_sample_cursor: int,
    world_size: int,
) -> DistributedStepControl:
    if not _should_continue_training_iterations(
        iteration_index=iteration_index, num_iterations_limit=num_iterations_limit
    ):
        return DistributedStepControl(
            opcode=_DISTRIBUTED_CONTROL_STOP,
            requested_batch_size=0,
            global_sample_cursor=global_sample_cursor,
            iteration_index=iteration_index,
        )

    remaining_samples = sample_count - global_sample_cursor
    max_feasible_batch_size = remaining_samples // world_size
    if max_feasible_batch_size <= 0:
        return DistributedStepControl(
            opcode=_DISTRIBUTED_CONTROL_SKIP,
            requested_batch_size=0,
            global_sample_cursor=0,
            iteration_index=iteration_index + 1,
        )

    return DistributedStepControl(
        opcode=_DISTRIBUTED_CONTROL_RUN,
        requested_batch_size=1,
        global_sample_cursor=global_sample_cursor,
        iteration_index=iteration_index,
    )


def _maybe_emit_master_progress(
    *,
    clock: TrainingLoopClock,
    samples_trained: int,
    iteration_index: int,
    num_iterations_limit: int,
) -> None:
    now = time.monotonic()
    if (not _is_primary_rank()) or (now - clock.last_master_progress_time < 1.0):
        return
    elapsed = _elapsed_training_time_sec(clock=clock, now=now)
    progress = min(1.0, float(iteration_index) / float(num_iterations_limit))
    label = (
        f"Training: {samples_trained} samples trained "
        f"(iteration {iteration_index}/{num_iterations_limit}, elapsed {elapsed:.1f}s)"
    )
    _text_master_progress(progress, label)
    clock.last_master_progress_time = now


def _compute_lr_multiplier(
    *,
    step_index: int,
    warmup_steps: int,
    min_lr_scale: float,
) -> float:
    assert step_index >= 0, "step_index must be non-negative"
    assert warmup_steps >= 0, "warmup_steps must be non-negative"
    assert min_lr_scale > 0.0 and min_lr_scale <= 1.0, "min_lr_scale must be in (0, 1]"

    # --- warmup phase (linear ramp from min_lr_scale to 1.0) ---
    if warmup_steps > 0 and step_index < warmup_steps:
        warmup_scale = float(step_index + 1) / float(warmup_steps)
        return max(min_lr_scale, min(1.0, warmup_scale))

    return 1.0



def _set_optimizer_learning_rate(
    *,
    optimizer: torch.optim.Optimizer,
    base_learning_rate: float,
    step_index: int,
    warmup_steps: int,
    min_lr_scale: float,
) -> float:
    multiplier = _compute_lr_multiplier(
        step_index=step_index,
        warmup_steps=warmup_steps,
        min_lr_scale=min_lr_scale,
    )
    current_learning_rate = base_learning_rate * multiplier
    for param_group in optimizer.param_groups:
        param_group["lr"] = current_learning_rate
    return current_learning_rate


def _maybe_clip_gradients(
    *, model: torch.nn.Module, max_grad_norm: float
) -> float | None:
    if max_grad_norm <= 0.0:
        return None
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    if isinstance(grad_norm, torch.Tensor):
        return float(grad_norm.detach().item())
    return float(grad_norm)


def _flush_partial_gradients(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    accumulation_step: int,
    global_step: int,
    max_grad_norm: float,
    base_learning_rate: float,
    lr_warmup_steps: int,
    lr_min_scale: float,
    is_distributed: bool,
) -> tuple[int, int, float | None, float]:
    current_lr = float(optimizer.param_groups[0]["lr"])
    if accumulation_step <= 0:
        return accumulation_step, global_step, None, current_lr
    if is_distributed:
        optimizer.zero_grad(set_to_none=True)
        if _is_primary_rank():
            _text_warning(
                "discarding_partial_gradients=1 "
                "reason=distributed_unsynced_final_microbatch "
                f"accumulation_step={accumulation_step} global_step={global_step}"
            )
        return 0, global_step, None, current_lr
    clipped_grad_norm = _maybe_clip_gradients(model=model, max_grad_norm=max_grad_norm)
    _assert_pre_step_finite(model, optimizer, clipped_grad_norm)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    next_global_step = global_step + 1
    current_lr = _set_optimizer_learning_rate(
        optimizer=optimizer,
        base_learning_rate=base_learning_rate,
        step_index=next_global_step,
        warmup_steps=lr_warmup_steps,
        min_lr_scale=lr_min_scale,
    )
    return 0, next_global_step, clipped_grad_norm, current_lr


def _finalize_training_run(
    *,
    rank: int,
    global_step: int,
    samples_trained: int,
    training_time: float,
    final_model_output_parent_dir: Path,
    eng: Any,
) -> None:
    eng._distributed_barrier()
    _text_delete_worker_bar(f"rank{rank}")
    if _is_primary_rank():
        _text_info(
            f"finished_training=1 global_step={global_step} "
            f"samples_trained={samples_trained} "
            f"training_time={training_time:.1f}s "
            f"final_model_output_parent_dir={final_model_output_parent_dir}"
        )
    eng._shutdown_distributed_process_group()





def _sample_average_supervised_advantage(
    sample: TrainingSampleTokenized,
) -> float:
    assert len(sample.input_ids) == len(sample.labels), (
        "input_ids and labels must align"
    )
    assert len(sample.token_advantages) == len(sample.input_ids), (
        "token_advantages and input_ids must align"
    )
    supervised_advantages = [
        advantage
        for label, advantage in zip(sample.labels, sample.token_advantages)
        if label != IGNORE_LABEL
    ]
    assert len(supervised_advantages) > 0, "sample must contain supervised tokens"
    return float(statistics.fmean(supervised_advantages))


def _compute_abs_advantage_stats_for_available_samples(
    *, lazy_loader: LazyResolvedBatchLoader
) -> tuple[float, float, float]:
    assert lazy_loader.sample_count > 0, "sample_count must be positive"
    _memtrace(
        "compute_abs_advantage_stats_begin "
        f"sample_count={lazy_loader.sample_count}"
    )
    absolute_advantages: list[float] = []
    for sample_index in range(lazy_loader.sample_count):
        sample = lazy_loader.get_sample(sample_index)
        absolute_advantages.append(abs(_sample_average_supervised_advantage(sample)))
        if sample_index == 0:
            _memtrace(
                "compute_abs_advantage_stats_first_sample "
                f"sample_index={sample_index} input_length={sample.input_length}"
            )
    assert len(absolute_advantages) > 0, "absolute_advantages cannot be empty"
    max_abs_advantage = max(absolute_advantages)
    min_abs_advantage = min(absolute_advantages)
    median_abs_advantage = float(statistics.median(absolute_advantages))
    _memtrace(
        "compute_abs_advantage_stats_end "
        f"sample_count={lazy_loader.sample_count} "
        f"max_abs_advantage={max_abs_advantage:.6f} "
        f"min_abs_advantage={min_abs_advantage:.6f} "
        f"median_abs_advantage={median_abs_advantage:.6f}"
    )
    return max_abs_advantage, min_abs_advantage, median_abs_advantage


def _write_training_summary(
    *,
    training_summary_parent_dir: str,
    samples_available: int,
    samples_trained: int,
    samples_trained_this_run: int,
    global_step: int,
    min_grad_accum_steps: int,
    max_average_absolute_advantage: float,
    min_average_absolute_advantage: float,
    median_average_absolute_advantage: float,
    total_training_time_sec: float,
    longest_non_oom_trajectory_length: int,
    stopped_due_to_oom: bool,
) -> None:
    assert len(training_summary_parent_dir.strip()) > 0, (
        "training_summary_parent_dir cannot be empty"
    )
    assert samples_available > 0, "samples_available must be positive"
    assert samples_trained >= 0, "samples_trained must be non-negative"
    assert samples_trained_this_run >= 0, "samples_trained_this_run must be non-negative"
    assert total_training_time_sec >= 0.0, (
        "total_training_time_sec must be non-negative"
    )
    assert longest_non_oom_trajectory_length >= 0, (
        "longest_non_oom_trajectory_length must be non-negative"
    )
    iterations = float(samples_trained) / float(samples_available)
    average_batch_size = 1.0
    if global_step > 0:
        average_batch_size = float(samples_trained) / float(global_step)
    payload = {
        "samples_available": int(samples_available),
        "samples_trained": int(samples_trained),
        "samples_trained_this_run": int(samples_trained_this_run),
        "iterations": float(iterations),
        "training_iterations_trained_cumulative": float(iterations),
        "global_step": int(global_step),
        "average_batch_size": float(average_batch_size),
        "max_average_absolute_advantage": float(max_average_absolute_advantage),
        "min_average_absolute_advantage": float(min_average_absolute_advantage),
        "median_average_absolute_advantage": float(median_average_absolute_advantage),
        "total_training_time_sec": float(total_training_time_sec),
        "longest_non_oom_trajectory_length": int(longest_non_oom_trajectory_length),
        "stopped_due_to_oom": bool(stopped_due_to_oom),
    }
    output_parent = Path(training_summary_parent_dir)
    output_parent.mkdir(parents=True, exist_ok=True)
    output_path = output_parent / "training_summary.json"
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _planned_question_rounded_accumulation_steps(
    *,
    lazy_loader: LazyResolvedBatchLoader,
    start_sample_index: int,
    min_grad_accum_steps: int,
) -> int:
    assert min_grad_accum_steps > 0, "min_grad_accum_steps must be positive"
    assert 0 <= start_sample_index < lazy_loader.sample_count, (
        "start_sample_index out of range"
    )
    planned_steps = 0
    last_question_id: int | None = None
    for sample_index in range(start_sample_index, lazy_loader.sample_count):
        sample = lazy_loader.get_sample(sample_index)
        planned_steps += 1
        last_question_id = sample.id.question_id
        next_sample_index = sample_index + 1
        if planned_steps < min_grad_accum_steps:
            continue
        if next_sample_index >= lazy_loader.sample_count:
            return planned_steps
        next_question_id = lazy_loader.get_sample(next_sample_index).id.question_id
        if next_question_id != last_question_id:
            return planned_steps
    assert planned_steps > 0, "planned accumulation window cannot be empty"
    return planned_steps


def _planned_question_rounded_accumulation_samples(
    *,
    lazy_loader: LazyResolvedBatchLoader,
    start_sample_index: int,
    min_samples: int,
) -> int:
    assert min_samples > 0, "min_samples must be positive"
    assert 0 <= start_sample_index < lazy_loader.sample_count, (
        "start_sample_index out of range"
    )
    planned_samples = 0
    last_question_id: int | None = None
    for sample_index in range(start_sample_index, lazy_loader.sample_count):
        sample = lazy_loader.get_sample(sample_index)
        planned_samples += 1
        last_question_id = sample.id.question_id
        next_sample_index = sample_index + 1
        if planned_samples < min_samples:
            continue
        if next_sample_index >= lazy_loader.sample_count:
            return planned_samples
        next_question_id = lazy_loader.get_sample(next_sample_index).id.question_id
        if next_question_id != last_question_id:
            return planned_samples
    assert planned_samples > 0, "planned accumulation window cannot be empty"
    return planned_samples


def _run_unified_loop(
    *,
    config: Any,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lazy_loader: LazyResolvedBatchLoader,
    resume_state: Any,
    rank: int,
    world_size: int,
    device: torch.device,
    pad_token_id: int,
    eos_token_id: int,
    bos_token_id: int,
    model_vocab_size: int,
    expected_model_name: str,
    logs_path: Path,
    final_model_output_parent_dir: Path,
    training_summary_parent_dir: str,
    resolved_model_path: str,
    source_model_path_for_save: str | None,
    tokenizer: Any,
    max_grad_norm: float,
    lr_warmup_steps: int,
    lr_min_scale: float,
    eng: Any,
    finalize_training: bool = True,
) -> Any:
    assert lazy_loader.sample_count > 0, "training set must be non-empty"
    lora_or_full = assert_supported_lora_or_full(config.lora_or_full)
    distributed_strategy = assert_supported_distributed_strategy(config.distributed_strategy)
    is_distributed = world_size > 1
    fail_fast_on_cuda_oom = _should_fail_fast_on_cuda_oom(
        distributed_strategy=distributed_strategy,
        world_size=world_size,
        training_set_sort_mode=str(getattr(config, "training_set_sort_mode", "")),
    )
    if is_distributed:
        assert lazy_loader.sample_count >= world_size, (
            "sample_count must be >= world_size for distributed training"
        )

    trajectory_length_cap = _maybe_apply_single_gpu_lora_synthetic_oom_preflight(
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
        resume_state=resume_state,
        eng=eng,
    )

    sample_count = lazy_loader.sample_count

    # Cap warmup steps so the LR reaches its peak before the dataset is exhausted
    # in a single pass (critical for tiny datasets).
    if lr_warmup_steps > 0 and sample_count > 0:
        max_warmup_steps = max(1, sample_count // (2 * config.min_grad_accum_steps))
        if lr_warmup_steps > max_warmup_steps:
            if _is_primary_rank():
                _text_info(
                    "lr_warmup_capped_for_dataset_size=1 "
                    f"original_warmup_steps={lr_warmup_steps} "
                    f"capped_warmup_steps={max_warmup_steps} "
                    f"sample_count={sample_count} "
                    f"grad_accum_steps={config.min_grad_accum_steps}"
                )
            lr_warmup_steps = max_warmup_steps

    if _is_primary_rank():
        _text_info(
            "lr_warmup_config=1 "
            f"lr_warmup_steps={lr_warmup_steps} "
            f"lr_min_scale={lr_min_scale:.4f}"
        )

    samples_available = sample_count
    if (
        resume_state.samples_available == samples_available
        and resume_state.max_average_absolute_advantage >= 0.0
        and resume_state.min_average_absolute_advantage >= 0.0
        and resume_state.median_average_absolute_advantage >= 0.0
    ):
        max_average_absolute_advantage = float(
            resume_state.max_average_absolute_advantage
        )
        min_average_absolute_advantage = float(
            resume_state.min_average_absolute_advantage
        )
        median_average_absolute_advantage = float(
            resume_state.median_average_absolute_advantage
        )
    else:
        (
            max_average_absolute_advantage,
            min_average_absolute_advantage,
            median_average_absolute_advantage,
        ) = _compute_abs_advantage_stats_for_available_samples(lazy_loader=lazy_loader)

    global_sample_cursor = resume_state.next_sample_index
    if global_sample_cursor >= sample_count:
        global_sample_cursor = 0

    max_input_token_id = -1
    max_label_token_id = -1
    data_model_name = expected_model_name

    _emit_trajectory_length_cap(cap=trajectory_length_cap)
    if _is_primary_rank():
        _text_info(
            "training_set_advantage_stats=1 "
            f"samples_available={samples_available} "
            f"max_average_absolute_advantage={max_average_absolute_advantage:.6f} "
            f"min_average_absolute_advantage={min_average_absolute_advantage:.6f} "
            f"median_average_absolute_advantage={median_average_absolute_advantage:.6f}"
        )
        _text_info(
            "tokenizer special tokens "
            f"pad_token_id={pad_token_id} eos_token_id={eos_token_id} bos_token_id={bos_token_id}"
        )
        eng._log_json_line(
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
            },
        )

    global_step = resume_state.global_step
    accumulation_step = 0
    if resume_state.accumulation_step != 0 and _is_primary_rank():
        _text_warning(
            "discarding_partial_accumulation_after_resume=1 "
            f"resume_accumulation_step={resume_state.accumulation_step}"
        )
    current_learning_rate = _set_optimizer_learning_rate(
        optimizer=optimizer,
        base_learning_rate=config.learning_rate,
        step_index=global_step,
        warmup_steps=lr_warmup_steps,
        min_lr_scale=lr_min_scale,
    )
    resumed_elapsed_training_time_sec = min(
        config.training_time, max(0.0, resume_state.elapsed_training_time_sec)
    )
    clock = _init_training_loop_clock(
        training_time=config.training_time,
        resumed_elapsed_training_time_sec=resumed_elapsed_training_time_sec,
    )
    samples_trained = max(0, int(resume_state.samples_trained))
    samples_trained_at_loop_start = samples_trained
    longest_non_oom_trajectory_length = max(
        0, int(getattr(resume_state, "longest_non_oom_trajectory_length", 0))
    )
    stopped_due_to_oom = False
    optimizer.zero_grad(set_to_none=True)
    current_accumulation_target = max(1, int(config.min_grad_accum_steps))
    current_accumulation_target_samples = current_accumulation_target
    accumulation_samples = 0
    max_batch_tokens = max(0, int(getattr(config, "max_batch_tokens", 0)))
    if _is_primary_rank():
        _text_info(
            "dynamic_batching_config=1 "
            f"max_batch_tokens={max_batch_tokens} "
            f"single_gpu_enabled={int((not is_distributed) and max_batch_tokens > 0)}"
        )

    iteration_index = max(0, resume_state.next_iteration_index)

    while True:
        _maybe_emit_master_progress(
            clock=clock,
            samples_trained=samples_trained,
            iteration_index=iteration_index,
            num_iterations_limit=config.num_iterations_limit,
        )

        if is_distributed:
            control_tensor = torch.zeros(4, dtype=torch.int64, device=device)
            if _is_primary_rank():
                control = _plan_distributed_step_control(
                    num_iterations_limit=config.num_iterations_limit,
                    iteration_index=iteration_index,
                    sample_count=sample_count,
                    global_sample_cursor=global_sample_cursor,
                    world_size=world_size,
                )
                global_sample_cursor = control.global_sample_cursor
                iteration_index = control.iteration_index
                control_tensor[0] = control.opcode
                control_tensor[1] = control.requested_batch_size
                control_tensor[2] = global_sample_cursor
                control_tensor[3] = iteration_index

            torch.distributed.broadcast(control_tensor, src=0)
            control_opcode = int(control_tensor[0].item())
            requested_batch_size = int(control_tensor[1].item())
            global_sample_cursor = int(control_tensor[2].item())
            iteration_index = int(control_tensor[3].item())
            if control_opcode == _DISTRIBUTED_CONTROL_STOP:
                break
            if control_opcode != _DISTRIBUTED_CONTROL_RUN:
                continue

            rank_sample_start = global_sample_cursor + (rank * requested_batch_size)
            rank_sample_end = rank_sample_start + requested_batch_size
            assert rank_sample_end <= sample_count, (
                "rank interval must be within sample range"
            )
            worker_progress = float(rank_sample_start) / sample_count
            _text_worker_progress(
                f"rank{rank}",
                worker_progress,
                f"Sample {rank_sample_start}/{sample_count}",
            )

            window = lazy_loader.resolve_batch(
                sample_index=rank_sample_start,
                batch_size=requested_batch_size,
                batch_index=rank_sample_start,
            )
        else:
            if not _should_continue_training_iterations(
                iteration_index=iteration_index,
                num_iterations_limit=config.num_iterations_limit,
            ):
                break
            remaining_samples = sample_count - global_sample_cursor
            if remaining_samples <= 0:
                global_sample_cursor = 0
                iteration_index += 1
                continue
            if accumulation_step == 0:
                current_accumulation_target_samples = (
                    _planned_question_rounded_accumulation_samples(
                        lazy_loader=lazy_loader,
                        start_sample_index=global_sample_cursor,
                        min_samples=config.min_grad_accum_steps,
                    )
                )
                if (
                    current_accumulation_target_samples > 2 * config.min_grad_accum_steps
                    and _is_primary_rank()
                ):
                    _text_warning(
                        "large_question_rounded_accumulation_window=1 "
                        f"min_grad_accum_steps={config.min_grad_accum_steps} "
                        f"planned_accumulation_samples={current_accumulation_target_samples} "
                        f"start_sample_index={global_sample_cursor}"
                    )
            samples_needed_for_accumulation = max(
                1, current_accumulation_target_samples - accumulation_samples
            )
            requested_batch_size = min(samples_needed_for_accumulation, remaining_samples)

            worker_progress = float(global_sample_cursor) / sample_count
            _text_worker_progress(
                f"rank{rank}",
                worker_progress,
                f"Sample {global_sample_cursor}/{sample_count}",
            )
            if max_batch_tokens > 0:
                window = lazy_loader.resolve_token_budget_batch(
                    sample_index=global_sample_cursor,
                    max_batch_tokens=max_batch_tokens,
                    max_batch_size=requested_batch_size,
                    batch_index=global_sample_cursor,
                )
            else:
                window = lazy_loader.resolve_batch(
                    sample_index=global_sample_cursor,
                    batch_size=min(1, requested_batch_size),
                    batch_index=global_sample_cursor,
                )

        resolved_batch = window.resolved_batch
        step_start_global_sample_cursor = global_sample_cursor
        step_start_iteration_index = iteration_index
        step_start_samples_trained = samples_trained
        step_samples = _truncate_samples_to_cap(
            resolved_batch.samples, trajectory_length_cap
        )
        step_batch = ResolvedTrainingBatch(
            batch_index=resolved_batch.batch_index,
            ids=resolved_batch.ids,
            samples=step_samples,
            model_official_name=resolved_batch.model_official_name,
        )

        for sample in step_batch.samples:
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
        assert max_input_token_id < model_vocab_size, (
            "input_ids contain token id out of model vocab range"
        )
        assert max_label_token_id < model_vocab_size, (
            "labels contain token id out of model vocab range"
        )

        if not is_distributed and max_batch_tokens <= 0 and accumulation_step == 0:
            current_accumulation_target = _planned_question_rounded_accumulation_steps(
                lazy_loader=lazy_loader,
                start_sample_index=global_sample_cursor,
                min_grad_accum_steps=config.min_grad_accum_steps,
            )
            if (
                current_accumulation_target > 2 * config.min_grad_accum_steps
                and _is_primary_rank()
            ):
                _text_warning(
                    "large_question_rounded_accumulation_window=1 "
                    f"min_grad_accum_steps={config.min_grad_accum_steps} "
                    f"planned_accumulation_steps={current_accumulation_target} "
                    f"start_sample_index={global_sample_cursor}"
                )
        elif is_distributed:
            current_accumulation_target = config.min_grad_accum_steps

        if torch.cuda.is_available() and device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device=device)
        step_start = time.perf_counter()
        if not is_distributed and max_batch_tokens > 0:
            should_sync = (
                accumulation_samples + len(step_batch.samples)
            ) >= current_accumulation_target_samples
        else:
            should_sync = (accumulation_step + 1) == current_accumulation_target
        sync_context: AbstractContextManager[None] = nullcontext()
        no_sync_method = getattr(model, "no_sync", None)
        if is_distributed and callable(no_sync_method) and not should_sync:
            sync_context = cast(AbstractContextManager[None], no_sync_method())
        collated = None
        input_ids = None
        labels = None
        attention_mask = None
        advantages = None
        old_logprobs = None
        ref_logprobs = None
        logits = None
        loss_output = None
        loss = None
        batch_token_length = max(sample.input_length for sample in step_batch.samples)
        emit_step_memtrace = _is_primary_rank() and (
            step_batch.batch_index < 3 or step_start_samples_trained == 0
        )
        try:
            if emit_step_memtrace:
                _memtrace(
                    "step_pre_collate "
                    f"batch_index={step_batch.batch_index} "
                    f"global_sample_cursor={global_sample_cursor} "
                    f"requested_batch_size={requested_batch_size} "
                    f"batch_token_length={batch_token_length}"
                )
            collated = collate_training_samples(
                samples=step_batch.samples, pad_token_id=pad_token_id
            )
            if emit_step_memtrace:
                _memtrace(
                    "step_post_collate "
                    f"batch_index={step_batch.batch_index} "
                    f"input_ids_shape={tuple(collated.input_ids.shape)} "
                    f"labels_shape={tuple(collated.labels.shape)} "
                    f"advantages_shape={tuple(collated.advantages.shape)} "
                    f"old_logprobs_shape={tuple(collated.old_logprobs.shape)} "
                    f"ref_logprobs_shape={tuple(collated.ref_logprobs.shape) if collated.ref_logprobs is not None else None}"
                )
                _memtrace(f"step_pre_device_copy batch_index={step_batch.batch_index}")
            input_ids = collated.input_ids.to(device=device, non_blocking=True)
            labels = collated.labels.to(device=device, non_blocking=True)
            attention_mask = collated.attention_mask.to(
                device=device, non_blocking=True
            )
            advantages = collated.advantages.to(device=device, non_blocking=True)
            old_logprobs = collated.old_logprobs.to(device=device, non_blocking=True)
            ref_logprobs = (
                collated.ref_logprobs.to(device=device, non_blocking=True)
                if collated.ref_logprobs is not None
                else None
            )
            if emit_step_memtrace:
                _memtrace(
                    "step_post_device_copy "
                    f"batch_index={step_batch.batch_index} device={device}"
                )

            with sync_context:
                if emit_step_memtrace:
                    _memtrace(f"step_pre_forward batch_index={step_batch.batch_index}")
                logits = eng._forward_logits(
                    model, input_ids=input_ids, attention_mask=attention_mask
                )
                if emit_step_memtrace:
                    _memtrace(
                        "step_post_forward "
                        f"batch_index={step_batch.batch_index} "
                        f"logits_shape={tuple(logits.shape)}"
                    )
                loss_output = compute_advantage_weighted_causal_lm_loss(
                    logits=logits,
                    labels=labels,
                    advantages=advantages,
                    old_logprobs=old_logprobs,
                    ref_logprobs=ref_logprobs,
                    advantage_clip=config.advantage_clip,
                    kl_beta=config.kl_beta,
                )
                if not is_distributed and max_batch_tokens > 0:
                    loss = loss_output.loss * (
                        float(len(step_batch.samples))
                        / float(current_accumulation_target_samples)
                    )
                else:
                    loss = loss_output.loss / current_accumulation_target
                if emit_step_memtrace:
                    _memtrace(
                        "step_pre_backward "
                        f"batch_index={step_batch.batch_index} "
                        f"loss={float(loss.detach().item()):.8f}"
                    )
                loss.backward()
                if emit_step_memtrace:
                    _memtrace(f"step_post_backward batch_index={step_batch.batch_index}")
        except (RuntimeError, AssertionError) as exc:
            if eng._is_cuda_oom_exception(exc):
                if is_distributed:
                    raise RuntimeError(
                        "CUDA OOM encountered during distributed training; "
                        "graceful OOM recovery is only supported for single-process training"
                    ) from exc
                collated = None
                input_ids = None
                labels = None
                attention_mask = None
                advantages = None
                old_logprobs = None
                ref_logprobs = None
                logits = None
                loss_output = None
                loss = None
                samples_trained = step_start_samples_trained
                global_sample_cursor = step_start_global_sample_cursor
                iteration_index = step_start_iteration_index
                accumulation_step = 0
                accumulation_samples = 0
                eng._print_cuda_oom_diagnostics_stderr(
                    rank=rank,
                    iteration_index=iteration_index,
                    batch_index=step_batch.batch_index,
                    device=device,
                )
                optimizer.zero_grad(set_to_none=True)
                eng._release_step_memory(device)
                loop_samples_trained = samples_trained - samples_trained_at_loop_start
                if loop_samples_trained <= 0:
                    raise RuntimeError(
                        "CUDA OOM encountered before any training samples completed; "
                        "rerun with a shorter trajectory length or smaller model"
                    ) from exc
                if fail_fast_on_cuda_oom:
                    stopped_due_to_oom = True
                    eng._print_cuda_oom_stderr(
                        rank=rank,
                        iteration_index=iteration_index,
                        batch_index=step_batch.batch_index,
                        batch_token_length=batch_token_length,
                        next_batch_size=requested_batch_size,
                        will_retry=False,
                    )
                    _text_warning(
                        "cuda_oom_fail_fast=1 "
                        f"distributed_strategy={distributed_strategy} "
                        f"world_size={world_size} "
                        f"iteration={iteration_index} "
                        f"batch_index={step_batch.batch_index} "
                        f"batch_token_length={batch_token_length} "
                        f"samples_trained_this_run={loop_samples_trained}"
                    )
                    break
                oom_plan = _plan_cuda_oom(iteration_index=iteration_index)
                assert oom_plan.stop_training
                stopped_due_to_oom = True
                eng._print_cuda_oom_stderr(
                    rank=rank,
                    iteration_index=iteration_index,
                    batch_index=step_batch.batch_index,
                    batch_token_length=batch_token_length,
                    next_batch_size=requested_batch_size,
                    will_retry=False,
                )
                _text_warning(
                    "cuda_oom_stop_training=1 "
                    f"iteration={iteration_index} "
                    f"batch_index={step_batch.batch_index} "
                    f"batch_token_length={batch_token_length} "
                    f"samples_trained_this_run={loop_samples_trained}"
                )
                break
            if not eng._is_nonfinite_logits_exception(exc):
                if _is_primary_rank():
                    backward_diagnostics = " ".join(
                        [
                            _tensor_diagnostic_fragment("input_ids", input_ids),
                            _tensor_diagnostic_fragment("labels", labels),
                            _tensor_diagnostic_fragment(
                                "attention_mask", attention_mask
                            ),
                            _tensor_diagnostic_fragment("advantages", advantages),
                            _tensor_diagnostic_fragment("logits", logits),
                        ]
                    )
                    _text_warning(
                        "unexpected_backward_failure=1 "
                        f"error_type={type(exc).__name__} "
                        f"error_message={exc} "
                        f"step={global_step} iteration={iteration_index} "
                        f"batch_index={step_batch.batch_index} "
                        f"requested_batch_size={requested_batch_size} "
                        f"{backward_diagnostics}"
                    )
                raise

            nonfinite_tensor_name, nonfinite_nan_count, nonfinite_inf_count = (
                eng._parse_nonfinite_tensor_exception(exc)
            )
            nonfinite_forward_trace = None
            if (
                nonfinite_tensor_name == "logits"
                and input_ids is not None
                and attention_mask is not None
            ):
                logits = None
                loss_output = None
                loss = None
                collated = None
                labels = None
                advantages = None
                old_logprobs = None
                ref_logprobs = None
                eng._release_step_memory(device)
                try:
                    nonfinite_forward_trace = eng.trace_first_nonfinite_forward_module(
                        model_engine=model,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )
                except Exception as trace_exc:
                    nonfinite_forward_trace = (
                        "nonfinite_forward_trace=1 status=diagnostic_failed "
                        f"error_type={type(trace_exc).__name__}"
                    )
            collated = None
            input_ids = None
            labels = None
            attention_mask = None
            advantages = None
            old_logprobs = None
            ref_logprobs = None
            logits = None
            loss_output = None
            loss = None
            eng._release_step_memory(device)
            batch_token_length = max(
                sample.input_length for sample in step_batch.samples
            )
            nonfinite_trace_extra = _extract_nonfinite_trace_suffix(str(exc))
            if nonfinite_forward_trace is not None:
                nonfinite_trace_extra = f" {nonfinite_forward_trace}"
            _text_error(
                f"rank={rank} "
                "nonfinite=1 "
                f"nonfinite_tensor={nonfinite_tensor_name} "
                f"nonfinite_nan_count={nonfinite_nan_count} "
                f"nonfinite_inf_count={nonfinite_inf_count} "
                f"batch_index={step_batch.batch_index} "
                f"batch_token_length={batch_token_length} "
                f"sample_index={global_sample_cursor}"
                f"{nonfinite_trace_extra}"
            )
            if _is_primary_rank():
                eng._log_json_line(
                    logs_path,
                    {
                        "step": global_step,
                        "iteration": iteration_index,
                        "batch_index": step_batch.batch_index,
                        "nonfinite": 1,
                        "nonfinite_abort": 1,
                        "nonfinite_tensor_code": _nonfinite_tensor_code(
                            nonfinite_tensor_name
                        ),
                        "nonfinite_nan_count": nonfinite_nan_count,
                        "nonfinite_inf_count": nonfinite_inf_count,
                    },
                )
            if torch.cuda.is_available() and device.type == "cuda":
                torch.cuda.synchronize(device=device)
            raise RuntimeError(
                f"nonfinite tensor detected: tensor={nonfinite_tensor_name} "
                f"nan_count={nonfinite_nan_count} inf_count={nonfinite_inf_count} "
                f"batch_index={step_batch.batch_index}"
                f"{nonfinite_trace_extra}"
            ) from exc

        longest_non_oom_trajectory_length = max(
            longest_non_oom_trajectory_length, batch_token_length
        )

        step_elapsed_sec = max(time.perf_counter() - step_start, 1e-6)
        throughput_samples_per_sec = float(len(step_batch.samples)) / step_elapsed_sec
        gpu_memory_usage_pct = 100.0 * eng._gpu_memory_utilization(device)
        if torch.cuda.is_available() and device.type == "cuda":
            torch.cuda.synchronize(device=device)
        gpu_memory_allocated_pct = 100.0 * eng._gpu_memory_peak_allocated_ratio(device)
        gpu_memory_reserved_pct = 100.0 * eng._gpu_memory_reserved_ratio(device)
        _text_key_value(
            "throughput_samples_per_sec", f"{throughput_samples_per_sec:.2f}"
        )
        _text_key_value("batch_size", str(len(step_batch.samples)))
        _text_key_value("batch_token_length", str(int(input_ids.shape[1])))
        _text_key_value("requested_batch_size", str(requested_batch_size))
        _text_key_value("trajectory_length_cap", str(trajectory_length_cap))
        _text_key_value("gpu_memory_usage_pct", f"{gpu_memory_usage_pct:.2f}")
        _text_key_value("gpu_memory_allocated_pct", f"{gpu_memory_allocated_pct:.2f}")
        _text_key_value("gpu_memory_reserved_pct", f"{gpu_memory_reserved_pct:.2f}")
        if _is_primary_rank():
            _text_key_value("global_step", str(global_step))
            _text_key_value("iteration", str(iteration_index))
            _text_key_value("batch_index", str(step_batch.batch_index))
            _text_key_value("learning_rate", f"{current_learning_rate:.10f}")
            for stat_key, stat_value in loss_output.stats.items():
                _text_key_value(stat_key, f"{stat_value:.6f}")

        accumulation_step += 1
        accumulation_samples += len(step_batch.samples)
        if is_distributed:
            samples_trained += requested_batch_size * world_size
        else:
            samples_trained += len(step_batch.samples)
        if is_distributed:
            global_sample_cursor += requested_batch_size * world_size
            if global_sample_cursor >= sample_count:
                global_sample_cursor = 0
                iteration_index += 1
        else:
            global_sample_cursor = window.next_sample_index
            if global_sample_cursor >= sample_count:
                global_sample_cursor = 0
                iteration_index += 1

        reached_accumulation_target = (
            accumulation_samples >= current_accumulation_target_samples
            if (not is_distributed and max_batch_tokens > 0)
            else accumulation_step == current_accumulation_target
        )
        if reached_accumulation_target:
            try:
                clipped_grad_norm = _maybe_clip_gradients(
                    model=model, max_grad_norm=max_grad_norm
                )
                _assert_pre_step_finite(model, optimizer, clipped_grad_norm)
                optimizer.step()
            except (RuntimeError, AssertionError) as exc:
                if isinstance(exc, RuntimeError) and (
                    not eng._is_cuda_oom_exception(exc)
                ):
                    raise
                if isinstance(exc, AssertionError):
                    nonfinite_details = eng._parse_nonfinite_tensor_exception(exc)
                    if nonfinite_details is None:
                        raise
                    nonfinite_tensor_name, nonfinite_nan_count, nonfinite_inf_count = (
                        nonfinite_details
                    )
                    nonfinite_backward_trace = None
                    if (
                        nonfinite_tensor_name in {"gradients", "grad_norm"}
                        and input_ids is not None
                        and attention_mask is not None
                and labels is not None
                and advantages is not None
                and old_logprobs is not None
            ):
                        optimizer.zero_grad(set_to_none=True)
                        eng._release_step_memory(device)
                        try:
                            nonfinite_backward_trace = (
                                trace_first_nonfinite_backward_signal(
                                    model=model,
                                    input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    labels=labels,
                                    advantages=advantages,
                                    old_logprobs=old_logprobs,
                                    ref_logprobs=ref_logprobs,
                                    advantage_clip=config.advantage_clip,
                                    kl_beta=config.kl_beta,
                                    min_grad_accum_steps=config.min_grad_accum_steps,
                                )
                            )
                        except Exception as trace_exc:
                            nonfinite_backward_trace = (
                                "nonfinite_backward_trace=1 status=diagnostic_failed "
                                f"error_type={type(trace_exc).__name__}"
                            )
                        finally:
                            optimizer.zero_grad(set_to_none=True)
                    nonfinite_trace_extra = _extract_nonfinite_trace_suffix(str(exc))
                    if nonfinite_backward_trace is not None:
                        nonfinite_trace_extra = f" {nonfinite_backward_trace}"
                    _text_error(
                        f"rank={rank} nonfinite=1 "
                        f"nonfinite_tensor={nonfinite_tensor_name} "
                        f"nonfinite_nan_count={nonfinite_nan_count} "
                        f"nonfinite_inf_count={nonfinite_inf_count} "
                        f"batch_index={step_batch.batch_index} "
                        f"sample_index={global_sample_cursor} "
                        f"step={global_step} iteration={iteration_index}"
                        f"{nonfinite_trace_extra}"
                    )
                    if _is_primary_rank():
                        eng._log_json_line(
                            logs_path,
                            {
                                "step": global_step,
                                "iteration": iteration_index,
                                "batch_index": step_batch.batch_index,
                                "nonfinite": 1,
                                "nonfinite_abort": 1,
                                "nonfinite_tensor_code": _nonfinite_tensor_code(
                                    nonfinite_tensor_name
                                ),
                                "nonfinite_nan_count": nonfinite_nan_count,
                                "nonfinite_inf_count": nonfinite_inf_count,
                            },
                        )
                    raise RuntimeError(
                        f"nonfinite tensor detected before optimizer step: tensor={nonfinite_tensor_name} "
                        f"nan_count={nonfinite_nan_count} inf_count={nonfinite_inf_count} "
                        f"batch_index={step_batch.batch_index}"
                        f"{nonfinite_trace_extra}"
                    ) from exc
                clipped_grad_norm = None
                if is_distributed:
                    raise RuntimeError(
                        "CUDA OOM encountered during distributed training; "
                        "graceful OOM recovery is only supported for single-process training"
                    ) from exc
                samples_trained = step_start_samples_trained
                global_sample_cursor = step_start_global_sample_cursor
                iteration_index = step_start_iteration_index
                accumulation_step = 0
                accumulation_samples = 0
                eng._print_cuda_oom_diagnostics_stderr(
                    rank=rank,
                    iteration_index=iteration_index,
                    batch_index=step_batch.batch_index,
                    device=device,
                )
                optimizer.zero_grad(set_to_none=True)
                eng._release_step_memory(device)
                loop_samples_trained = samples_trained - samples_trained_at_loop_start
                if loop_samples_trained <= 0:
                    raise RuntimeError(
                        "CUDA OOM encountered before any training samples completed; "
                        "rerun with a shorter trajectory length or smaller model"
                    ) from exc
                if fail_fast_on_cuda_oom:
                    stopped_due_to_oom = True
                    eng._print_cuda_oom_stderr(
                        rank=rank,
                        iteration_index=iteration_index,
                        batch_index=step_batch.batch_index,
                        batch_token_length=batch_token_length,
                        next_batch_size=requested_batch_size,
                        will_retry=False,
                    )
                    _text_warning(
                        "cuda_oom_fail_fast=1 "
                        f"distributed_strategy={distributed_strategy} "
                        f"world_size={world_size} "
                        f"iteration={iteration_index} "
                        f"batch_index={step_batch.batch_index} "
                        f"batch_token_length={batch_token_length} "
                        f"samples_trained_this_run={loop_samples_trained}"
                    )
                    break
                oom_plan = _plan_cuda_oom(iteration_index=iteration_index)
                assert oom_plan.stop_training
                stopped_due_to_oom = True
                eng._print_cuda_oom_stderr(
                    rank=rank,
                    iteration_index=iteration_index,
                    batch_index=step_batch.batch_index,
                    batch_token_length=batch_token_length,
                    next_batch_size=requested_batch_size,
                    will_retry=False,
                )
                _text_warning(
                    "cuda_oom_stop_training=1 "
                    f"iteration={iteration_index} "
                    f"batch_index={step_batch.batch_index} "
                    f"batch_token_length={batch_token_length} "
                    f"samples_trained_this_run={loop_samples_trained}"
                )
                break
            optimizer.zero_grad(set_to_none=True)
            accumulation_step = 0
            accumulation_samples = 0
            current_accumulation_target = config.min_grad_accum_steps
            current_accumulation_target_samples = config.min_grad_accum_steps
            global_step += 1
            current_learning_rate = _set_optimizer_learning_rate(
                optimizer=optimizer,
                base_learning_rate=config.learning_rate,
                step_index=global_step,
                warmup_steps=lr_warmup_steps,
                min_lr_scale=lr_min_scale,
            )

            now = time.monotonic()
            elapsed_since_last_log_sec = now - clock.last_log_time
            if (
                _is_primary_rank()
                and elapsed_since_last_log_sec >= config.log_time_interval
            ):
                log_payload: dict[str, float | int] = {
                    "step": global_step,
                    "iteration": iteration_index,
                    "batch_index": step_batch.batch_index,
                    "actual_batch_size": len(step_batch.samples),
                    "step_time_sec": float(step_elapsed_sec),
                    "throughput_samples_per_sec": throughput_samples_per_sec,
                    "gpu_memory_usage_pct": gpu_memory_usage_pct,
                    "trajectory_length_cap": trajectory_length_cap,
                    "learning_rate": float(current_learning_rate),
                }
                if clipped_grad_norm is not None:
                    log_payload["grad_norm"] = float(clipped_grad_norm)
                for key, value in loss_output.stats.items():
                    log_payload[key] = value
                eng._log_json_line(logs_path, log_payload)
                clock.last_log_time = now

    accumulation_step, global_step, clipped_grad_norm, current_learning_rate = (
        _flush_partial_gradients(
            model=model,
            optimizer=optimizer,
            accumulation_step=accumulation_step,
            global_step=global_step,
            max_grad_norm=max_grad_norm,
            base_learning_rate=config.learning_rate,
            lr_warmup_steps=lr_warmup_steps,
            lr_min_scale=lr_min_scale,
            is_distributed=is_distributed,
        )
    )
    if _is_primary_rank() and clipped_grad_norm is not None:
        _text_key_value("flush_grad_norm", f"{clipped_grad_norm:.6f}")
    if _is_primary_rank():
        _text_key_value("learning_rate", f"{current_learning_rate:.10f}")

    total_training_time_sec = _elapsed_training_time_sec(clock=clock)
    if _is_primary_rank():
        _memtrace(
            "pre_final_model_save "
            f"final_model_output_parent_dir={final_model_output_parent_dir}"
        )
    eng._save_final_model_folder(
        model=model,
        lora_or_full=lora_or_full,
        distributed_strategy=distributed_strategy,
        final_model_output_parent_dir=final_model_output_parent_dir,
        source_model_path=source_model_path_for_save or resolved_model_path,
        tokenizer=tokenizer,
        lora_save_mode=getattr(config, "lora_save_mode", "adapter"),
    )
    if _is_primary_rank():
        _memtrace(
            "post_final_model_save "
            f"final_model_output_parent_dir={final_model_output_parent_dir}"
        )
    if _is_primary_rank():
        _text_info(
            f"final_model_saved=1 final_model_output_parent_dir={final_model_output_parent_dir}"
        )
    if _is_primary_rank():
        _write_training_summary(
            training_summary_parent_dir=training_summary_parent_dir,
            samples_available=samples_available,
            samples_trained=samples_trained,
            samples_trained_this_run=samples_trained - samples_trained_at_loop_start,
            global_step=global_step,
            min_grad_accum_steps=config.min_grad_accum_steps,
            max_average_absolute_advantage=max_average_absolute_advantage,
            min_average_absolute_advantage=min_average_absolute_advantage,
            median_average_absolute_advantage=median_average_absolute_advantage,
            total_training_time_sec=total_training_time_sec,
            longest_non_oom_trajectory_length=longest_non_oom_trajectory_length,
            stopped_due_to_oom=stopped_due_to_oom,
        )
        _text_info(
            "training_summary_written=1 "
            f"samples_trained={samples_trained} "
            f"global_step={global_step} "
            f"training_summary_parent_dir={training_summary_parent_dir}"
        )
    final_resume_state = eng.ResumeState(
        global_step=global_step,
        next_iteration_index=iteration_index,
        next_batch_cursor=global_sample_cursor,
        accumulation_step=accumulation_step,
        next_sample_index=global_sample_cursor,
        elapsed_training_time_sec=total_training_time_sec,
        samples_trained=samples_trained,
        samples_available=samples_available,
        max_average_absolute_advantage=max_average_absolute_advantage,
        min_average_absolute_advantage=min_average_absolute_advantage,
        median_average_absolute_advantage=median_average_absolute_advantage,
        samples_trained_this_run=samples_trained - samples_trained_at_loop_start,
        longest_non_oom_trajectory_length=longest_non_oom_trajectory_length,
        stopped_due_to_oom=stopped_due_to_oom,
    )
    if finalize_training:
        _finalize_training_run(
            rank=rank,
            global_step=global_step,
            samples_trained=samples_trained,
            training_time=config.training_time,
            final_model_output_parent_dir=final_model_output_parent_dir,
            eng=eng,
        )
    else:
        eng._distributed_barrier()
    return final_resume_state


def run_training_loop(
    *,
    config: Any,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    resume_state: Any,
    rank: int,
    world_size: int,
    device: torch.device,
    pad_token_id: int,
    eos_token_id: int,
    bos_token_id: int,
    model_vocab_size: int,
    expected_model_name: str,
    logs_path: Path,
    final_model_output_parent_dir: Path,
    training_summary_parent_dir: str,
    resolved_model_path: str,
    source_model_path_for_save: str | None,
    tokenizer: Any,
    max_grad_norm: float,
    lr_warmup_steps: int,
    lr_min_scale: float,
    lazy_loader: LazyResolvedBatchLoader | None,
) -> None:
    from . import engine as eng

    assert lazy_loader is not None, "lazy_loader is required for training"
    _run_unified_loop(
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
        final_model_output_parent_dir=final_model_output_parent_dir,
        training_summary_parent_dir=training_summary_parent_dir,
        resolved_model_path=resolved_model_path,
        source_model_path_for_save=source_model_path_for_save,
        tokenizer=tokenizer,
        max_grad_norm=max_grad_norm,
        lr_warmup_steps=lr_warmup_steps,
        lr_min_scale=lr_min_scale,
        eng=eng,
    )
