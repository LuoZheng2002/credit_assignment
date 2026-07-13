from __future__ import annotations

import json
import math
import statistics
import time
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch

from ..tui_logging import (
    _tui_delete_worker_bar,
    _tui_error,
    _tui_info,
    _tui_key_value,
    _tui_master_progress,
    _tui_warning,
    _tui_worker_progress,
)
from .batch_dataset import LazyResolvedBatchLoader, ResolvedTrainingBatch
from .collator import IGNORE_LABEL, collate_training_samples
from .data_msgpack import TrainingSampleTokenized
from .losses import compute_advantage_weighted_causal_lm_loss
from .training_plan import (
    assert_supported_distributed_strategy,
    assert_supported_lora_or_full,
)


@dataclass
class TrainingLoopClock:
    training_time: float
    resumed_elapsed_training_time_sec: float
    run_start_time: float
    training_end_time: float
    last_checkpoint_save_time: float
    last_log_time: float
    last_master_progress_time: float


@dataclass(frozen=True)
class DistributedStepControl:
    opcode: int
    requested_batch_size: int
    global_sample_cursor: int
    iteration_index: int



DEFAULT_TRAJECTORY_LENGTH_CAP = 4096
MIN_TRAJECTORY_LENGTH_CAP = 2
_DISTRIBUTED_CONTROL_STOP = -1
_DISTRIBUTED_CONTROL_SKIP = 0
_DISTRIBUTED_CONTROL_RUN = 1


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
    return TrainingSampleTokenized(
        id=sample.id,
        input_ids=truncated_input_ids,
        labels=truncated_labels,
        input_length=len(truncated_input_ids),
        token_advantages=truncated_token_advantages,
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
    _tui_key_value("trajectory_length_cap", str(cap))


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
    advantage_clip: float,
    grad_accum_steps: int,
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
            advantage_clip=advantage_clip,
        )
        replay_loss = loss_output.loss / grad_accum_steps
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
    _assert_gradient_tensors_finite(model)
    if clipped_grad_norm is not None:
        _assert_scalar_finite(
            clipped_grad_norm,
            "grad_norm",
            f"value={clipped_grad_norm}",
        )
    _assert_optimizer_state_finite(model, optimizer)


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
        training_end_time=run_start_time
        + max(0.0, training_time - resumed_elapsed_training_time_sec),
        last_checkpoint_save_time=run_start_time,
        last_log_time=run_start_time,
        last_master_progress_time=run_start_time - 1.0,
    )


def _should_continue_training(
    *, clock: TrainingLoopClock, iteration_index: int, num_iterations_limit: int
) -> bool:
    return (
        time.monotonic() < clock.training_end_time
        and iteration_index < num_iterations_limit
    )


def _elapsed_training_time_sec(
    *, clock: TrainingLoopClock, now: float | None = None
) -> float:
    if now is None:
        now = time.monotonic()
    elapsed = clock.resumed_elapsed_training_time_sec + (now - clock.run_start_time)
    return min(clock.training_time, elapsed)


def _plan_distributed_step_control(
    *,
    clock: TrainingLoopClock,
    iteration_index: int,
    num_iterations_limit: int,
    sample_count: int,
    global_sample_cursor: int,
    world_size: int,
) -> DistributedStepControl:
    if not _should_continue_training(
        clock=clock,
        iteration_index=iteration_index,
        num_iterations_limit=num_iterations_limit,
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
    *, clock: TrainingLoopClock, samples_trained: int
) -> None:
    now = time.monotonic()
    if (not _is_primary_rank()) or (now - clock.last_master_progress_time < 1.0):
        return
    elapsed = _elapsed_training_time_sec(clock=clock, now=now)
    progress = min(1.0, elapsed / clock.training_time)
    label = f"Training: {samples_trained} samples trained ({elapsed:.1f}s/{clock.training_time:.1f}s)"
    _tui_master_progress(progress, label)
    clock.last_master_progress_time = now


LR_SCHEDULE_SQRT = "sqrt"
LR_SCHEDULE_COSINE = "cosine"


def _compute_lr_multiplier(
    *,
    step_index: int,
    warmup_steps: int,
    min_lr_scale: float,
    schedule: str,
    total_steps: int,
) -> float:
    assert step_index >= 0, "step_index must be non-negative"
    assert warmup_steps >= 0, "warmup_steps must be non-negative"
    assert min_lr_scale > 0.0 and min_lr_scale <= 1.0, "min_lr_scale must be in (0, 1]"
    assert schedule in {LR_SCHEDULE_SQRT, LR_SCHEDULE_COSINE}, (
        f"lr_schedule must be one of: {LR_SCHEDULE_SQRT}, {LR_SCHEDULE_COSINE}"
    )
    assert total_steps >= 0, "total_steps must be non-negative"

    # --- warmup phase (linear ramp from min_lr_scale to 1.0) ---
    if warmup_steps > 0 and step_index < warmup_steps:
        warmup_scale = float(step_index + 1) / float(warmup_steps)
        return max(min_lr_scale, min(1.0, warmup_scale))

    if warmup_steps <= 0:
        return 1.0

    # --- decay phase ---
    if schedule == LR_SCHEDULE_COSINE and total_steps > warmup_steps:
        decay_steps = total_steps - warmup_steps
        progress = min(1.0, float(step_index - warmup_steps) / float(decay_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_scale + (1.0 - min_lr_scale) * cosine_decay

    # sqrt decay (original behaviour, also used as fallback when total_steps is
    # not meaningfully larger than warmup_steps for cosine)
    decay_scale = math.sqrt(float(warmup_steps) / float(max(step_index, warmup_steps)))
    return max(min_lr_scale, min(1.0, decay_scale))


def _set_optimizer_learning_rate(
    *,
    optimizer: torch.optim.Optimizer,
    base_learning_rate: float,
    step_index: int,
    warmup_steps: int,
    min_lr_scale: float,
    schedule: str,
    total_steps: int,
) -> float:
    multiplier = _compute_lr_multiplier(
        step_index=step_index,
        warmup_steps=warmup_steps,
        min_lr_scale=min_lr_scale,
        schedule=schedule,
        total_steps=total_steps,
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
    lr_schedule: str,
    lr_total_steps: int,
    is_distributed: bool,
) -> tuple[int, int, float | None, float]:
    current_lr = float(optimizer.param_groups[0]["lr"])
    if accumulation_step <= 0:
        return accumulation_step, global_step, None, current_lr
    if is_distributed:
        optimizer.zero_grad(set_to_none=True)
        if _is_primary_rank():
            _tui_warning(
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
        schedule=lr_schedule,
        total_steps=lr_total_steps,
    )
    return 0, next_global_step, clipped_grad_norm, current_lr


def _finalize_training_run(
    *,
    rank: int,
    global_step: int,
    training_time: float,
    final_model_output_parent_dir: Path,
    eng: Any,
) -> None:
    eng._distributed_barrier()
    _tui_delete_worker_bar(f"rank{rank}")
    if _is_primary_rank():
        _tui_info(
            f"finished_training=1 global_step={global_step} "
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
    absolute_advantages: list[float] = []
    for sample_index in range(lazy_loader.sample_count):
        sample = lazy_loader.get_sample(sample_index)
        absolute_advantages.append(abs(_sample_average_supervised_advantage(sample)))
    assert len(absolute_advantages) > 0, "absolute_advantages cannot be empty"
    max_abs_advantage = max(absolute_advantages)
    min_abs_advantage = min(absolute_advantages)
    median_abs_advantage = float(statistics.median(absolute_advantages))
    return max_abs_advantage, min_abs_advantage, median_abs_advantage


def _write_training_summary(
    *,
    training_summary_parent_dir: str,
    samples_available: int,
    samples_trained: int,
    max_average_absolute_advantage: float,
    min_average_absolute_advantage: float,
    median_average_absolute_advantage: float,
    total_training_time_sec: float,
) -> None:
    assert len(training_summary_parent_dir.strip()) > 0, (
        "training_summary_parent_dir cannot be empty"
    )
    assert samples_available > 0, "samples_available must be positive"
    assert samples_trained >= 0, "samples_trained must be non-negative"
    assert total_training_time_sec >= 0.0, (
        "total_training_time_sec must be non-negative"
    )
    iterations = float(samples_trained) / float(samples_available)
    payload = {
        "samples_available": int(samples_available),
        "samples_trained": int(samples_trained),
        "iterations": float(iterations),
        "max_average_absolute_advantage": float(max_average_absolute_advantage),
        "min_average_absolute_advantage": float(min_average_absolute_advantage),
        "median_average_absolute_advantage": float(median_average_absolute_advantage),
        "total_training_time_sec": float(total_training_time_sec),
    }
    output_parent = Path(training_summary_parent_dir)
    output_parent.mkdir(parents=True, exist_ok=True)
    output_path = output_parent / "training_summary.json"
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


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
    if is_distributed:
        assert lazy_loader.sample_count >= world_size, (
            "sample_count must be >= world_size for distributed training"
        )

    trajectory_length_cap = DEFAULT_TRAJECTORY_LENGTH_CAP

    sample_count = lazy_loader.sample_count

    lr_schedule = config.lr_schedule
    assert lr_schedule in {LR_SCHEDULE_SQRT, LR_SCHEDULE_COSINE}, (
        f"lr_schedule must be one of: {LR_SCHEDULE_SQRT}, {LR_SCHEDULE_COSINE}"
    )

    # Cap warmup steps so the LR reaches its peak before the dataset is exhausted
    # in a single pass (critical for tiny datasets).
    if lr_warmup_steps > 0 and sample_count > 0:
        max_warmup_steps = max(1, sample_count // (2 * config.grad_accum_steps))
        if lr_warmup_steps > max_warmup_steps:
            if _is_primary_rank():
                _tui_info(
                    "lr_warmup_capped_for_dataset_size=1 "
                    f"original_warmup_steps={lr_warmup_steps} "
                    f"capped_warmup_steps={max_warmup_steps} "
                    f"sample_count={sample_count} "
                    f"grad_accum_steps={config.grad_accum_steps}"
                )
            lr_warmup_steps = max_warmup_steps

    # Auto-compute total decay steps when not explicitly provided.
    # Conservative estimate: assume batch_size=1 and each sample is visited
    # num_iterations_limit times, then divide by grad_accum_steps.
    if config.lr_total_steps > 0:
        lr_total_steps = config.lr_total_steps
    else:
        lr_total_steps = max(
            1,
            (sample_count * config.num_iterations_limit) // config.grad_accum_steps,
        )

    if _is_primary_rank():
        _tui_info(
            "lr_schedule_config=1 "
            f"lr_schedule={lr_schedule} "
            f"lr_warmup_steps={lr_warmup_steps} "
            f"lr_total_steps={lr_total_steps} "
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
        _tui_info(
            "training_set_advantage_stats=1 "
            f"samples_available={samples_available} "
            f"max_average_absolute_advantage={max_average_absolute_advantage:.6f} "
            f"min_average_absolute_advantage={min_average_absolute_advantage:.6f} "
            f"median_average_absolute_advantage={median_average_absolute_advantage:.6f}"
        )
        _tui_info(
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
    accumulation_step = resume_state.accumulation_step
    current_learning_rate = _set_optimizer_learning_rate(
        optimizer=optimizer,
        base_learning_rate=config.learning_rate,
        step_index=global_step,
        warmup_steps=lr_warmup_steps,
        min_lr_scale=lr_min_scale,
        schedule=lr_schedule,
        total_steps=lr_total_steps,
    )
    resumed_elapsed_training_time_sec = min(
        config.training_time, max(0.0, resume_state.elapsed_training_time_sec)
    )
    clock = _init_training_loop_clock(
        training_time=config.training_time,
        resumed_elapsed_training_time_sec=resumed_elapsed_training_time_sec,
    )
    samples_trained = max(0, int(resume_state.samples_trained))
    optimizer.zero_grad(set_to_none=True)

    iteration_index = max(0, resume_state.next_iteration_index)

    while True:
        _maybe_emit_master_progress(clock=clock, samples_trained=samples_trained)

        if is_distributed:
            control_tensor = torch.zeros(4, dtype=torch.int64, device=device)
            if _is_primary_rank():
                control = _plan_distributed_step_control(
                    clock=clock,
                    iteration_index=iteration_index,
                    num_iterations_limit=config.num_iterations_limit,
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
            _tui_worker_progress(
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
            if not _should_continue_training(
                clock=clock,
                iteration_index=iteration_index,
                num_iterations_limit=config.num_iterations_limit,
            ):
                break
            requested_batch_size = min(1, sample_count - global_sample_cursor)
            if requested_batch_size <= 0:
                global_sample_cursor = 0
                iteration_index += 1
                continue

            worker_progress = float(global_sample_cursor) / sample_count
            _tui_worker_progress(
                f"rank{rank}",
                worker_progress,
                f"Sample {global_sample_cursor}/{sample_count}",
            )
            window = lazy_loader.resolve_batch(
                sample_index=global_sample_cursor,
                batch_size=requested_batch_size,
                batch_index=global_sample_cursor,
            )

        resolved_batch = window.resolved_batch
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

        if torch.cuda.is_available() and device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device=device)
        step_start = time.perf_counter()
        should_sync = (accumulation_step + 1) == config.grad_accum_steps
        sync_context: AbstractContextManager[None] = nullcontext()
        no_sync_method = getattr(model, "no_sync", None)
        if is_distributed and callable(no_sync_method) and not should_sync:
            sync_context = cast(AbstractContextManager[None], no_sync_method())
        collated = None
        input_ids = None
        labels = None
        attention_mask = None
        advantages = None
        logits = None
        loss_output = None
        loss = None
        try:
            collated = collate_training_samples(
                samples=step_batch.samples, pad_token_id=pad_token_id
            )
            input_ids = collated.input_ids.to(device=device, non_blocking=True)
            labels = collated.labels.to(device=device, non_blocking=True)
            attention_mask = collated.attention_mask.to(
                device=device, non_blocking=True
            )
            advantages = collated.advantages.to(device=device, non_blocking=True)

            with sync_context:
                logits = eng._forward_logits(
                    model, input_ids=input_ids, attention_mask=attention_mask
                )
                loss_output = compute_advantage_weighted_causal_lm_loss(
                    logits=logits,
                    labels=labels,
                    advantages=advantages,
                    advantage_clip=config.advantage_clip,
                )
                loss = loss_output.loss / config.grad_accum_steps
                loss.backward()
                try:
                    _assert_gradient_tensors_finite(model)
                except AssertionError as grad_exc:
                    nonfinite_details = eng._parse_nonfinite_tensor_exception(grad_exc)
                    if nonfinite_details is None:
                        raise
                    optimizer.zero_grad(set_to_none=True)
                    eng._release_step_memory(device)
                    nonfinite_backward_trace = None
                    try:
                        nonfinite_backward_trace = (
                            trace_first_nonfinite_backward_signal(
                                model=model,
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                labels=labels,
                                advantages=advantages,
                                advantage_clip=config.advantage_clip,
                                grad_accum_steps=config.grad_accum_steps,
                            )
                        )
                    except Exception as trace_exc:
                        nonfinite_backward_trace = (
                            "nonfinite_backward_trace=1 status=diagnostic_failed "
                            f"error_type={type(trace_exc).__name__}"
                        )
                    finally:
                        optimizer.zero_grad(set_to_none=True)
                    raise AssertionError(
                        f"{grad_exc} accumulation_micro_step={accumulation_step + 1} "
                        f"batch_index={step_batch.batch_index} sample_index={global_sample_cursor}"
                        f" {nonfinite_backward_trace}"
                    ) from grad_exc
        except (RuntimeError, AssertionError) as exc:
            if eng._is_cuda_oom_exception(exc):
                collated = None
                input_ids = None
                labels = None
                attention_mask = None
                advantages = None
                logits = None
                loss_output = None
                loss = None
                eng._print_cuda_oom_diagnostics_stderr(
                    rank=rank,
                    iteration_index=iteration_index,
                    batch_index=step_batch.batch_index,
                    device=device,
                )
                eng._release_step_memory(device)
                if is_distributed:
                    raise RuntimeError(
                        "distributed CUDA OOM recovery is unsupported under FSDP/DDP because "
                        "ranks can diverge into mismatched collectives; rerun with a smaller fixed "
                        "batch size or shorter trajectory length"
                    ) from exc
                batch_token_length = max(
                    sample.input_length for sample in step_batch.samples
                )
                eng._print_cuda_oom_stderr(
                    rank=rank,
                    iteration_index=iteration_index,
                    batch_index=step_batch.batch_index,
                    batch_token_length=batch_token_length,
                    next_batch_size=1,
                    will_retry=False,
                )
                skipped_samples = requested_batch_size
                if _is_primary_rank():
                    _tui_warning(
                        "oom_at_batch_size_1=1 "
                        f"skipped_samples={skipped_samples} "
                        f"sample_index={global_sample_cursor}"
                    )
                    eng._log_json_line(
                        logs_path,
                        {
                            "step": global_step,
                            "iteration": iteration_index,
                            "batch_index": step_batch.batch_index,
                            "oom": 1,
                            "oom_skipped_sample": 1,
                            "skipped_samples": skipped_samples,
                        },
                    )
                global_sample_cursor += skipped_samples
                if torch.cuda.is_available() and device.type == "cuda":
                    torch.cuda.synchronize(device=device)
                continue
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
                    _tui_warning(
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
            _tui_error(
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

        step_elapsed_sec = max(time.perf_counter() - step_start, 1e-6)
        throughput_samples_per_sec = float(len(step_batch.samples)) / step_elapsed_sec
        gpu_memory_usage_pct = 100.0 * eng._gpu_memory_utilization(device)
        if torch.cuda.is_available() and device.type == "cuda":
            torch.cuda.synchronize(device=device)
        gpu_memory_allocated_pct = 100.0 * eng._gpu_memory_peak_allocated_ratio(device)
        gpu_memory_reserved_pct = 100.0 * eng._gpu_memory_reserved_ratio(device)
        _tui_key_value(
            "throughput_samples_per_sec", f"{throughput_samples_per_sec:.2f}"
        )
        _tui_key_value("batch_size", str(len(step_batch.samples)))
        _tui_key_value("batch_token_length", str(int(input_ids.shape[1])))
        _tui_key_value("requested_batch_size", str(requested_batch_size))
        _tui_key_value("trajectory_length_cap", str(trajectory_length_cap))
        _tui_key_value("gpu_memory_usage_pct", f"{gpu_memory_usage_pct:.2f}")
        _tui_key_value("gpu_memory_allocated_pct", f"{gpu_memory_allocated_pct:.2f}")
        _tui_key_value("gpu_memory_reserved_pct", f"{gpu_memory_reserved_pct:.2f}")
        if _is_primary_rank():
            _tui_key_value("global_step", str(global_step))
            _tui_key_value("iteration", str(iteration_index))
            _tui_key_value("batch_index", str(step_batch.batch_index))
            _tui_key_value("learning_rate", f"{current_learning_rate:.10f}")
            for stat_key, stat_value in loss_output.stats.items():
                _tui_key_value(stat_key, f"{stat_value:.6f}")

        accumulation_step += 1
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

        if accumulation_step == config.grad_accum_steps:
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
                                    advantage_clip=config.advantage_clip,
                                    grad_accum_steps=config.grad_accum_steps,
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
                    _tui_error(
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
                eng._print_cuda_oom_diagnostics_stderr(
                    rank=rank,
                    iteration_index=iteration_index,
                    batch_index=step_batch.batch_index,
                    device=device,
                )
                optimizer.zero_grad(set_to_none=True)
                eng._release_step_memory(device)
                if is_distributed:
                    raise RuntimeError(
                        "distributed CUDA OOM recovery is unsupported under FSDP/DDP because "
                        "ranks can diverge into mismatched collectives; rerun with a smaller fixed "
                        "batch size or shorter trajectory length"
                    ) from exc
                batch_token_length = max(
                    sample.input_length for sample in step_batch.samples
                )
                eng._print_cuda_oom_stderr(
                    rank=rank,
                    iteration_index=iteration_index,
                    batch_index=step_batch.batch_index,
                    batch_token_length=batch_token_length,
                    next_batch_size=1,
                    will_retry=False,
                )
                skipped_samples = requested_batch_size
                if _is_primary_rank():
                    _tui_warning(
                        "oom_at_batch_size_1=1 "
                        f"skipped_samples={skipped_samples} "
                        f"sample_index={global_sample_cursor}"
                    )
                    eng._log_json_line(
                        logs_path,
                        {
                            "step": global_step,
                            "iteration": iteration_index,
                            "batch_index": step_batch.batch_index,
                            "oom": 1,
                            "oom_skipped_sample": 1,
                            "skipped_samples": skipped_samples,
                        },
                    )
                global_sample_cursor += skipped_samples
                eng._release_step_memory(device)
                accumulation_step = 0
                if torch.cuda.is_available() and device.type == "cuda":
                    torch.cuda.synchronize(device=device)
                continue
            optimizer.zero_grad(set_to_none=True)
            accumulation_step = 0
            global_step += 1
            current_learning_rate = _set_optimizer_learning_rate(
                optimizer=optimizer,
                base_learning_rate=config.learning_rate,
                step_index=global_step,
                warmup_steps=lr_warmup_steps,
                min_lr_scale=lr_min_scale,
                schedule=lr_schedule,
                total_steps=lr_total_steps,
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
            lr_schedule=lr_schedule,
            lr_total_steps=lr_total_steps,
            is_distributed=is_distributed,
        )
    )
    if _is_primary_rank() and clipped_grad_norm is not None:
        _tui_key_value("flush_grad_norm", f"{clipped_grad_norm:.6f}")
    if _is_primary_rank():
        _tui_key_value("learning_rate", f"{current_learning_rate:.10f}")

    total_training_time_sec = _elapsed_training_time_sec(clock=clock)
    eng._save_final_model_folder(
        model=model,
        lora_or_full=lora_or_full,
        distributed_strategy=distributed_strategy,
        final_model_output_parent_dir=final_model_output_parent_dir,
        source_model_path=resolved_model_path,
        tokenizer=tokenizer,
    )
    if _is_primary_rank():
        _tui_info(
            f"final_model_saved=1 final_model_output_parent_dir={final_model_output_parent_dir}"
        )
    if _is_primary_rank():
        _write_training_summary(
            training_summary_parent_dir=training_summary_parent_dir,
            samples_available=samples_available,
            samples_trained=samples_trained,
            max_average_absolute_advantage=max_average_absolute_advantage,
            min_average_absolute_advantage=min_average_absolute_advantage,
            median_average_absolute_advantage=median_average_absolute_advantage,
            total_training_time_sec=total_training_time_sec,
        )
        _tui_info(
            "training_summary_written=1 "
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
    )
    if finalize_training:
        _finalize_training_run(
            rank=rank,
            global_step=global_step,
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
        tokenizer=tokenizer,
        max_grad_norm=max_grad_norm,
        lr_warmup_steps=lr_warmup_steps,
        lr_min_scale=lr_min_scale,
        eng=eng,
    )
