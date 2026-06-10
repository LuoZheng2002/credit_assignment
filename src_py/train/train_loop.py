from __future__ import annotations

import json
import math
import statistics
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from ..tui_logging import (
    _tui_delete_worker_bar,
    _tui_info,
    _tui_key_value,
    _tui_master_progress,
    _tui_warning,
    _tui_worker_progress,
)
from .batch_dataset import LazyResolvedBatchLoader, ResolvedTrainingBatch
from .collator import collate_training_samples
from .data_sqlite import TrainingSampleTokenized
from .losses import compute_advantage_weighted_causal_lm_loss
from .training_plan import assert_supported_training_plan


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
class AdaptiveBatchState:
    next_batch_size: int
    next_batch_size_float: float
    velocity: float
    throughput_ema: float
    best_throughput_ema: float
    memory_utilization_ema: float
    previous_tokens_per_sample: float


DEFAULT_TRAJECTORY_LENGTH_CAP = 4096
MIN_TRAJECTORY_LENGTH_CAP = 2


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
    return TrainingSampleTokenized(
        id=sample.id,
        input_ids=truncated_input_ids,
        labels=truncated_labels,
        reconstructed=sample.reconstructed,
        input_length=len(truncated_input_ids),
        advantage=sample.advantage,
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


def _is_primary_rank() -> bool:
    return (
        (not torch.distributed.is_available())
        or (not torch.distributed.is_initialized())
        or torch.distributed.get_rank() == 0
    )


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


def _compute_lr_multiplier(
    *, step_index: int, warmup_steps: int, min_lr_scale: float
) -> float:
    assert step_index >= 0, "step_index must be non-negative"
    assert warmup_steps >= 0, "warmup_steps must be non-negative"
    assert min_lr_scale > 0.0 and min_lr_scale <= 1.0, "min_lr_scale must be in (0, 1]"

    if warmup_steps > 0 and step_index < warmup_steps:
        warmup_scale = float(step_index + 1) / float(warmup_steps)
        return max(min_lr_scale, min(1.0, warmup_scale))

    if warmup_steps <= 0:
        return 1.0

    decay_scale = math.sqrt(float(warmup_steps) / float(max(step_index, warmup_steps)))
    return max(min_lr_scale, min(1.0, decay_scale))


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
) -> tuple[int, int, float | None, float]:
    if accumulation_step <= 0:
        current_lr = float(optimizer.param_groups[0]["lr"])
        return accumulation_step, global_step, None, current_lr
    clipped_grad_norm = _maybe_clip_gradients(model=model, max_grad_norm=max_grad_norm)
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


def _broadcast_adaptive_state_distributed(
    *, adaptive_state: AdaptiveBatchState
) -> AdaptiveBatchState:
    payload_box: list[dict[str, float | int]] = [{}]
    if _is_primary_rank():
        payload_box[0] = {
            "next_batch_size": int(adaptive_state.next_batch_size),
            "next_batch_size_float": float(adaptive_state.next_batch_size_float),
            "velocity": float(adaptive_state.velocity),
            "throughput_ema": float(adaptive_state.throughput_ema),
            "best_throughput_ema": float(adaptive_state.best_throughput_ema),
            "memory_utilization_ema": float(adaptive_state.memory_utilization_ema),
            "previous_tokens_per_sample": float(
                adaptive_state.previous_tokens_per_sample
            ),
        }
    torch.distributed.broadcast_object_list(payload_box, src=0)
    payload = payload_box[0]
    return AdaptiveBatchState(
        next_batch_size=int(payload["next_batch_size"]),
        next_batch_size_float=float(payload["next_batch_size_float"]),
        velocity=float(payload["velocity"]),
        throughput_ema=float(payload["throughput_ema"]),
        best_throughput_ema=float(payload["best_throughput_ema"]),
        memory_utilization_ema=float(payload["memory_utilization_ema"]),
        previous_tokens_per_sample=float(payload["previous_tokens_per_sample"]),
    )


def _compute_abs_advantage_stats_for_available_samples(
    *, lazy_loader: LazyResolvedBatchLoader
) -> tuple[float, float, float]:
    assert lazy_loader.sample_count > 0, "sample_count must be positive"
    absolute_advantages: list[float] = []
    for sample_index in range(lazy_loader.sample_count):
        sample = lazy_loader.get_sample(sample_index)
        absolute_advantages.append(abs(float(sample.advantage)))
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
    checkpoints_parent_dir: Path,
    final_model_output_parent_dir: Path,
    training_summary_parent_dir: str,
    resolved_model_path: str,
    tokenizer: Any,
    initial_batch_size: int,
    initial_adaptive_velocity: float,
    target_gpu_memory_utilization: float,
    max_batch_size_cap: int | None,
    reset_batch_size_on_wrap: bool,
    max_grad_norm: float,
    lr_warmup_steps: int,
    lr_min_scale: float,
    eng: Any,
) -> None:
    assert lazy_loader.sample_count > 0, "training set must be non-empty"
    training_plan = assert_supported_training_plan(config.training_plan)
    is_distributed = world_size > 1
    if is_distributed:
        assert lazy_loader.sample_count >= world_size, (
            "sample_count must be >= world_size for distributed training"
        )

    trajectory_length_cap = DEFAULT_TRAJECTORY_LENGTH_CAP

    sample_count = lazy_loader.sample_count
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

    distributed_batch_limit = (sample_count // world_size) if is_distributed else -1
    fallback_sample_index = max(0, resume_state.next_batch_cursor) * max(
        1, initial_batch_size
    )
    global_sample_cursor = resume_state.next_sample_index
    if global_sample_cursor == 0 and resume_state.next_batch_cursor > 0:
        global_sample_cursor = fallback_sample_index
    if global_sample_cursor >= sample_count:
        global_sample_cursor = 0

    adaptive_state = AdaptiveBatchState(
        next_batch_size=max(
            1,
            min(
                distributed_batch_limit if is_distributed else sample_count,
                resume_state.next_batch_size,
            ),
        ),
        next_batch_size_float=max(
            1.0,
            min(
                float(distributed_batch_limit if is_distributed else sample_count),
                resume_state.adaptive_next_batch_size_float
                if resume_state.adaptive_next_batch_size_float > 0.0
                else float(
                    max(
                        1,
                        min(
                            distributed_batch_limit if is_distributed else sample_count,
                            resume_state.next_batch_size,
                        ),
                    )
                ),
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

    if max_batch_size_cap is None:
        max_allowed_batch_size = (
            distributed_batch_limit if is_distributed else sample_count
        )
    else:
        max_allowed_batch_size = max(
            1,
            min(
                distributed_batch_limit if is_distributed else sample_count,
                max_batch_size_cap,
            ),
        )
    if adaptive_state.next_batch_size > max_allowed_batch_size:
        adaptive_state = AdaptiveBatchState(
            next_batch_size=max_allowed_batch_size,
            next_batch_size_float=float(max_allowed_batch_size),
            velocity=adaptive_state.velocity,
            throughput_ema=adaptive_state.throughput_ema,
            best_throughput_ema=adaptive_state.best_throughput_ema,
            memory_utilization_ema=adaptive_state.memory_utilization_ema,
            previous_tokens_per_sample=adaptive_state.previous_tokens_per_sample,
        )

    if _is_primary_rank() and max_batch_size_cap is not None:
        _tui_info(
            f"adaptive_batch_cap_active=1 max_batch_size={max_allowed_batch_size} "
            f"sample_count={sample_count}"
        )

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
                "local_batch_count": distributed_batch_limit if is_distributed else -1,
                "next_batch_size": adaptive_state.next_batch_size,
                "next_batch_size_int": adaptive_state.next_batch_size,
                "next_batch_size_float": adaptive_state.next_batch_size_float,
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

    while _should_continue_training(
        clock=clock,
        iteration_index=iteration_index,
        num_iterations_limit=config.num_iterations_limit,
    ):
        _maybe_emit_master_progress(
            clock=clock, samples_trained=samples_trained
        ), eng=eng
        )

        if is_distributed:
            control_tensor = torch.zeros(4, dtype=torch.int64, device=device)
            if _is_primary_rank():
                remaining_samples = sample_count - global_sample_cursor
                max_feasible_batch_size = remaining_samples // world_size
                if max_feasible_batch_size <= 0:
                    global_sample_cursor = 0
                    iteration_index += 1
                    if reset_batch_size_on_wrap:
                        adaptive_state = AdaptiveBatchState(
                            next_batch_size=1,
                            next_batch_size_float=1.0,
                            velocity=initial_adaptive_velocity,
                            throughput_ema=adaptive_state.throughput_ema,
                            best_throughput_ema=adaptive_state.best_throughput_ema,
                            memory_utilization_ema=adaptive_state.memory_utilization_ema,
                            previous_tokens_per_sample=adaptive_state.previous_tokens_per_sample,
                        )
                        _tui_info(
                            "adaptive_batch_wrap_reset_applied=1 "
                            f"iteration={iteration_index} next_batch_size={adaptive_state.next_batch_size}"
                        )
                    control_tensor[0] = 0
                else:
                    control_tensor[0] = 1
                    control_tensor[1] = max(
                        1, min(adaptive_state.next_batch_size, max_feasible_batch_size)
                    )
                control_tensor[2] = global_sample_cursor
                control_tensor[3] = iteration_index

            torch.distributed.broadcast(control_tensor, src=0)
            should_run_step = int(control_tensor[0].item()) == 1
            requested_batch_size = int(control_tensor[1].item())
            global_sample_cursor = int(control_tensor[2].item())
            iteration_index = int(control_tensor[3].item())
            if not should_run_step:
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
            requested_batch_size = min(
                adaptive_state.next_batch_size, sample_count - global_sample_cursor
            )
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
        sync_context = nullcontext()
        if is_distributed and hasattr(model, "no_sync") and not should_sync:
            sync_context = model.no_sync()
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
        except RuntimeError as exc:
            if not eng._is_cuda_oom_exception(exc):
                raise
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
            reduced_batch_size = max(1, adaptive_state.next_batch_size // 2)
            eng._print_cuda_oom_stderr(
                rank=rank,
                iteration_index=iteration_index,
                batch_index=step_batch.batch_index,
                batch_token_length=max(
                    sample.input_length for sample in step_batch.samples
                ),
                next_batch_size=(
                    adaptive_state.next_batch_size
                    if requested_batch_size <= 1
                    else reduced_batch_size
                ),
                will_retry=requested_batch_size > 1,
            )
            should_skip_sample = requested_batch_size <= 1
            if should_skip_sample:
                skipped_samples = (
                    requested_batch_size * world_size
                    if is_distributed
                    else requested_batch_size
                )
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
                            "next_batch_size": adaptive_state.next_batch_size,
                            "next_batch_size_float": adaptive_state.next_batch_size_float,
                        },
                    )
                global_sample_cursor += skipped_samples
            elif _is_primary_rank():
                adaptive_state = AdaptiveBatchState(
                    next_batch_size=reduced_batch_size,
                    next_batch_size_float=float(reduced_batch_size),
                    velocity=0.0,
                    throughput_ema=adaptive_state.throughput_ema,
                    best_throughput_ema=adaptive_state.best_throughput_ema,
                    memory_utilization_ema=adaptive_state.memory_utilization_ema,
                    previous_tokens_per_sample=adaptive_state.previous_tokens_per_sample,
                )
            if is_distributed:
                adaptive_state = _broadcast_adaptive_state_distributed(
                    adaptive_state=adaptive_state
                )
            if _is_primary_rank():
                eng._log_json_line(
                    logs_path,
                    {
                        "step": global_step,
                        "iteration": iteration_index,
                        "batch_index": step_batch.batch_index,
                        "oom": 1,
                        "next_batch_size": adaptive_state.next_batch_size,
                        "next_batch_size_float": adaptive_state.next_batch_size_float,
                    },
                )
            eng._release_step_memory(device)
            if torch.cuda.is_available() and device.type == "cuda":
                torch.cuda.synchronize(device=device)
            continue

        step_elapsed_sec = max(time.perf_counter() - step_start, 1e-6)
        throughput_samples_per_sec = float(len(step_batch.samples)) / step_elapsed_sec
        measured_tokens_per_sample = float(
            collated.attention_mask.sum(dim=1).float().mean().item()
        )
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
        _tui_key_value("next_batch_size", str(adaptive_state.next_batch_size))
        _tui_key_value("next_batch_size_int", str(adaptive_state.next_batch_size))
        _tui_key_value(
            "next_batch_size_float", f"{adaptive_state.next_batch_size_float:.2f}"
        )
        _tui_key_value("trajectory_length_cap", str(trajectory_length_cap))
        _tui_key_value("gpu_memory_usage_pct", f"{gpu_memory_usage_pct:.2f}")
        _tui_info(f"gpu_memory_usage_pct: {gpu_memory_usage_pct:.2f}%")
        _tui_key_value("gpu_memory_allocated_pct", f"{gpu_memory_allocated_pct:.2f}")
        _tui_info(f"gpu_memory_allocated_pct: {gpu_memory_allocated_pct:.2f}%")
        _tui_key_value("gpu_memory_reserved_pct", f"{gpu_memory_reserved_pct:.2f}")
        _tui_info(f"gpu_memory_reserved_pct: {gpu_memory_reserved_pct:.2f}%")
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
                if _is_primary_rank() and reset_batch_size_on_wrap:
                    adaptive_state = AdaptiveBatchState(
                        next_batch_size=1,
                        next_batch_size_float=1.0,
                        velocity=initial_adaptive_velocity,
                        throughput_ema=adaptive_state.throughput_ema,
                        best_throughput_ema=adaptive_state.best_throughput_ema,
                        memory_utilization_ema=adaptive_state.memory_utilization_ema,
                        previous_tokens_per_sample=adaptive_state.previous_tokens_per_sample,
                    )
                    _tui_info(
                        "adaptive_batch_wrap_reset_applied=1 "
                        f"iteration={iteration_index} next_batch_size={adaptive_state.next_batch_size}"
                    )
                if reset_batch_size_on_wrap:
                    adaptive_state = _broadcast_adaptive_state_distributed(
                        adaptive_state=adaptive_state
                    )
        else:
            global_sample_cursor = window.next_sample_index
            if global_sample_cursor >= sample_count:
                global_sample_cursor = 0
                iteration_index += 1
                if reset_batch_size_on_wrap:
                    adaptive_state = AdaptiveBatchState(
                        next_batch_size=1,
                        next_batch_size_float=1.0,
                        velocity=initial_adaptive_velocity,
                        throughput_ema=adaptive_state.throughput_ema,
                        best_throughput_ema=adaptive_state.best_throughput_ema,
                        memory_utilization_ema=adaptive_state.memory_utilization_ema,
                        previous_tokens_per_sample=adaptive_state.previous_tokens_per_sample,
                    )
                    if _is_primary_rank():
                        _tui_info(
                            "adaptive_batch_wrap_reset_applied=1 "
                            f"iteration={iteration_index} next_batch_size={adaptive_state.next_batch_size}"
                        )

        if accumulation_step == config.grad_accum_steps:
            clipped_grad_norm = _maybe_clip_gradients(
                model=model, max_grad_norm=max_grad_norm
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            accumulation_step = 0
            global_step += 1
            current_learning_rate = _set_optimizer_learning_rate(
                optimizer=optimizer,
                base_learning_rate=config.learning_rate,
                step_index=global_step,
                warmup_steps=lr_warmup_steps,
                min_lr_scale=lr_min_scale,
            )

            if (not is_distributed) or _is_primary_rank():
                adaptive_state = eng._update_adaptive_batch_state(
                    adaptive_state=adaptive_state,
                    measured_throughput=throughput_samples_per_sec,
                    measured_memory_utilization=gpu_memory_allocated_pct / 100.0,
                    measured_tokens_per_sample=measured_tokens_per_sample,
                    target_memory_utilization=target_gpu_memory_utilization,
                    min_batch_size=1,
                    max_batch_size=max_allowed_batch_size,
                )
            if is_distributed:
                adaptive_state = _broadcast_adaptive_state_distributed(
                    adaptive_state=adaptive_state
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
                    "next_batch_size": adaptive_state.next_batch_size,
                    "next_batch_size_float": adaptive_state.next_batch_size_float,
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

            elapsed_since_last_checkpoint_sec = now - clock.last_checkpoint_save_time
            if (
                elapsed_since_last_checkpoint_sec
                >= config.checkpoint_save_time_interval
            ):
                checkpoint_tag = "checkpoints"
                if _is_primary_rank():
                    _tui_info(
                        f"saving_periodic_checkpoint=1 elapsed_sec={elapsed_since_last_checkpoint_sec:.2f} "
                        f"global_step={global_step} iteration={iteration_index} "
                        f"batch_index={step_batch.batch_index} next_sample_index={global_sample_cursor}"
                    )
                eng._save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    output_dir=checkpoints_parent_dir,
                    checkpoint_tag=checkpoint_tag,
                    training_plan=training_plan,
                    global_step=global_step,
                    next_iteration_index=iteration_index,
                    next_batch_cursor=global_sample_cursor,
                    accumulation_step=accumulation_step,
                    next_sample_index=global_sample_cursor,
                    next_batch_size=adaptive_state.next_batch_size,
                    adaptive_velocity=adaptive_state.velocity,
                    adaptive_throughput_ema=adaptive_state.throughput_ema,
                    adaptive_best_throughput_ema=adaptive_state.best_throughput_ema,
                    adaptive_memory_utilization_ema=adaptive_state.memory_utilization_ema,
                    adaptive_previous_tokens_per_sample=adaptive_state.previous_tokens_per_sample,
                    adaptive_next_batch_size_float=adaptive_state.next_batch_size_float,
                    elapsed_training_time_sec=_elapsed_training_time_sec(
                        clock=clock, now=now
                    ),
                    samples_trained=samples_trained,
                    samples_available=samples_available,
                    max_average_absolute_advantage=max_average_absolute_advantage,
                    min_average_absolute_advantage=min_average_absolute_advantage,
                    median_average_absolute_advantage=median_average_absolute_advantage,
                )
                clock.last_checkpoint_save_time = now

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
        )
    )
    if _is_primary_rank() and clipped_grad_norm is not None:
        _tui_key_value("flush_grad_norm", f"{clipped_grad_norm:.6f}")
    if _is_primary_rank():
        _tui_key_value("learning_rate", f"{current_learning_rate:.10f}")

    total_training_time_sec = _elapsed_training_time_sec(clock=clock)
    eng._save_checkpoint(
        model=model,
        optimizer=optimizer,
        output_dir=checkpoints_parent_dir,
        checkpoint_tag="checkpoints",
        training_plan=training_plan,
        global_step=global_step,
        next_iteration_index=iteration_index,
        next_batch_cursor=global_sample_cursor,
        accumulation_step=accumulation_step,
        next_sample_index=global_sample_cursor,
        next_batch_size=adaptive_state.next_batch_size,
        adaptive_velocity=adaptive_state.velocity,
        adaptive_throughput_ema=adaptive_state.throughput_ema,
        adaptive_best_throughput_ema=adaptive_state.best_throughput_ema,
        adaptive_memory_utilization_ema=adaptive_state.memory_utilization_ema,
        adaptive_previous_tokens_per_sample=adaptive_state.previous_tokens_per_sample,
        adaptive_next_batch_size_float=adaptive_state.next_batch_size_float,
        elapsed_training_time_sec=total_training_time_sec,
        samples_trained=samples_trained,
        samples_available=samples_available,
        max_average_absolute_advantage=max_average_absolute_advantage,
        min_average_absolute_advantage=min_average_absolute_advantage,
        median_average_absolute_advantage=median_average_absolute_advantage,
    )
    eng._save_final_model_folder(
        model=model,
        training_plan=training_plan,
        final_model_output_parent_dir=final_model_output_parent_dir,
        source_model_path=resolved_model_path,
        tokenizer=tokenizer,
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
    _finalize_training_run(
        rank=rank,
        global_step=global_step,
        training_time=config.training_time,
        final_model_output_parent_dir=final_model_output_parent_dir,
        eng=eng,
    )


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
    checkpoints_parent_dir: Path,
    final_model_output_parent_dir: Path,
    training_summary_parent_dir: str,
    resolved_model_path: str,
    tokenizer: Any,
    initial_batch_size: int,
    initial_adaptive_velocity: float,
    target_gpu_memory_utilization: float,
    max_batch_size_cap: int | None,
    reset_batch_size_on_wrap: bool,
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
        checkpoints_parent_dir=checkpoints_parent_dir,
        final_model_output_parent_dir=final_model_output_parent_dir,
        training_summary_parent_dir=training_summary_parent_dir,
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
        eng=eng,
    )
