from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any

import torch

from .batch_dataset import LazyResolvedBatchLoader, ResolvedTrainingBatch
from .collator import collate_training_samples
from .data_sqlite import TrainingSampleTokenized
from .losses import compute_advantage_weighted_causal_lm_loss


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


def _reduce_trajectory_length_cap(current_cap: int) -> int:
    assert current_cap >= MIN_TRAJECTORY_LENGTH_CAP, "current_cap must be >= minimum"
    reduced = int(current_cap * 0.95)
    if reduced >= current_cap:
        reduced = current_cap - 1
    return max(MIN_TRAJECTORY_LENGTH_CAP, reduced)


def _truncate_sample_to_cap(sample: TrainingSampleTokenized, trajectory_length_cap: int) -> TrainingSampleTokenized:
    assert trajectory_length_cap >= MIN_TRAJECTORY_LENGTH_CAP, "trajectory_length_cap too small"
    if sample.input_length <= trajectory_length_cap:
        return sample

    start = sample.input_length - trajectory_length_cap
    truncated_input_ids = sample.input_ids[start:]
    truncated_labels = sample.labels[start:]
    assert len(truncated_input_ids) == len(truncated_labels), "truncated token arrays must align"
    assert len(truncated_input_ids) >= MIN_TRAJECTORY_LENGTH_CAP, "truncated sample is too short"
    return TrainingSampleTokenized(
        id=sample.id,
        input_ids=truncated_input_ids,
        labels=truncated_labels,
        reconstructed=sample.reconstructed,
        input_length=len(truncated_input_ids),
        advantage=sample.advantage,
        model_official_name=sample.model_official_name,
    )


def _truncate_samples_to_cap(samples: list[TrainingSampleTokenized], trajectory_length_cap: int) -> list[TrainingSampleTokenized]:
    assert len(samples) > 0, "samples cannot be empty"
    return [_truncate_sample_to_cap(sample, trajectory_length_cap) for sample in samples]


def _emit_trajectory_length_cap(*, cap: int, eng: Any) -> None:
    print(eng._json_key_value("trajectory_length_cap", str(cap)))


def _is_primary_rank() -> bool:
    return (not torch.distributed.is_available()) or (
        not torch.distributed.is_initialized()
    ) or torch.distributed.get_rank() == 0


def _init_training_loop_clock(*, training_time: float, resumed_elapsed_training_time_sec: float) -> TrainingLoopClock:
    run_start_time = time.monotonic()
    return TrainingLoopClock(
        training_time=training_time,
        resumed_elapsed_training_time_sec=resumed_elapsed_training_time_sec,
        run_start_time=run_start_time,
        training_end_time=run_start_time + max(0.0, training_time - resumed_elapsed_training_time_sec),
        last_checkpoint_save_time=run_start_time,
        last_log_time=run_start_time,
        last_master_progress_time=run_start_time - 1.0,
    )


def _should_continue_training(*, clock: TrainingLoopClock, iteration_index: int, num_iterations_limit: int) -> bool:
    return time.monotonic() < clock.training_end_time and iteration_index < num_iterations_limit


def _elapsed_training_time_sec(*, clock: TrainingLoopClock, now: float | None = None) -> float:
    if now is None:
        now = time.monotonic()
    elapsed = clock.resumed_elapsed_training_time_sec + (now - clock.run_start_time)
    return min(clock.training_time, elapsed)


def _maybe_emit_master_progress(*, clock: TrainingLoopClock, samples_trained: int, eng: Any) -> None:
    now = time.monotonic()
    if (not _is_primary_rank()) or (now - clock.last_master_progress_time < 1.0):
        return
    elapsed = _elapsed_training_time_sec(clock=clock, now=now)
    progress = min(1.0, elapsed / clock.training_time)
    label = f"Training: {samples_trained} samples trained ({elapsed:.1f}s/{clock.training_time:.1f}s)"
    print(eng._json_master_progress(progress, label))
    clock.last_master_progress_time = now


def _flush_partial_gradients(*, optimizer: torch.optim.Optimizer, accumulation_step: int, global_step: int) -> tuple[int, int]:
    if accumulation_step <= 0:
        return accumulation_step, global_step
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return 0, global_step + 1


def _finalize_training_run(*, rank: int, global_step: int, training_time: float, final_model_output_parent_dir: Path, eng: Any) -> None:
    eng._distributed_barrier()
    print(eng._json_delete_worker_bar(f"rank{rank}"))
    if _is_primary_rank():
        print(
            "[status] "
            f"finished_training=1 global_step={global_step} "
            f"training_time={training_time:.1f}s "
            f"final_model_output_parent_dir={final_model_output_parent_dir}"
        )
    eng._shutdown_distributed_process_group()


def _broadcast_adaptive_state_distributed(*, adaptive_state: AdaptiveBatchState) -> AdaptiveBatchState:
    payload_box: list[dict[str, float | int]] = [{}]
    if _is_primary_rank():
        payload_box[0] = {
            "next_batch_size": int(adaptive_state.next_batch_size),
            "next_batch_size_float": float(adaptive_state.next_batch_size_float),
            "velocity": float(adaptive_state.velocity),
            "throughput_ema": float(adaptive_state.throughput_ema),
            "best_throughput_ema": float(adaptive_state.best_throughput_ema),
            "memory_utilization_ema": float(adaptive_state.memory_utilization_ema),
            "previous_tokens_per_sample": float(adaptive_state.previous_tokens_per_sample),
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
    resolved_model_path: str,
    tokenizer: Any,
    initial_batch_size: int,
    initial_adaptive_velocity: float,
    target_gpu_memory_utilization: float,
    max_batch_size_cap: int | None,
    reset_batch_size_on_wrap: bool,
    eng: Any,
) -> None:
    assert lazy_loader.sample_count > 0, "training set must be non-empty"
    is_distributed = world_size > 1
    if is_distributed:
        assert lazy_loader.sample_count >= world_size, "sample_count must be >= world_size for distributed training"

    trajectory_length_cap = DEFAULT_TRAJECTORY_LENGTH_CAP

    sample_count = lazy_loader.sample_count
    distributed_batch_limit = (sample_count // world_size) if is_distributed else -1
    fallback_sample_index = max(0, resume_state.next_batch_cursor) * max(1, initial_batch_size)
    global_sample_cursor = resume_state.next_sample_index
    if global_sample_cursor == 0 and resume_state.next_batch_cursor > 0:
        global_sample_cursor = fallback_sample_index
    if global_sample_cursor >= sample_count:
        global_sample_cursor = 0

    adaptive_state = AdaptiveBatchState(
        next_batch_size=max(
            1,
            min(distributed_batch_limit if is_distributed else sample_count, resume_state.next_batch_size),
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
                        min(distributed_batch_limit if is_distributed else sample_count, resume_state.next_batch_size),
                    )
                ),
            ),
        ),
        velocity=(resume_state.adaptive_velocity if resume_state.adaptive_velocity > 0.0 else initial_adaptive_velocity),
        throughput_ema=resume_state.adaptive_throughput_ema,
        best_throughput_ema=resume_state.adaptive_best_throughput_ema,
        memory_utilization_ema=resume_state.adaptive_memory_utilization_ema,
        previous_tokens_per_sample=resume_state.adaptive_previous_tokens_per_sample,
    )

    if max_batch_size_cap is None:
        max_allowed_batch_size = distributed_batch_limit if is_distributed else sample_count
    else:
        max_allowed_batch_size = max(
            1,
            min(distributed_batch_limit if is_distributed else sample_count, max_batch_size_cap),
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
        print(
            "[status] "
            f"adaptive_batch_cap_active=1 max_batch_size={max_allowed_batch_size} "
            f"sample_count={sample_count}"
        )

    max_input_token_id = -1
    max_label_token_id = -1
    data_model_name = expected_model_name

    _emit_trajectory_length_cap(cap=trajectory_length_cap, eng=eng)
    if _is_primary_rank():
        print(
            "[startup] tokenizer special tokens "
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
    resumed_elapsed_training_time_sec = min(config.training_time, max(0.0, resume_state.elapsed_training_time_sec))
    clock = _init_training_loop_clock(
        training_time=config.training_time,
        resumed_elapsed_training_time_sec=resumed_elapsed_training_time_sec,
    )
    samples_trained = 0
    optimizer.zero_grad(set_to_none=True)

    iteration_index = max(0, resume_state.next_iteration_index)

    while _should_continue_training(clock=clock, iteration_index=iteration_index, num_iterations_limit=config.num_iterations_limit):
        _maybe_emit_master_progress(clock=clock, samples_trained=samples_trained, eng=eng)

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
                        print(
                            "[status] "
                            "adaptive_batch_wrap_reset_applied=1 "
                            f"iteration={iteration_index} next_batch_size={adaptive_state.next_batch_size}"
                        )
                    control_tensor[0] = 0
                else:
                    control_tensor[0] = 1
                    control_tensor[1] = max(1, min(adaptive_state.next_batch_size, max_feasible_batch_size))
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
            assert rank_sample_end <= sample_count, "rank interval must be within sample range"
            worker_progress = float(rank_sample_start) / sample_count
            print(eng._json_worker_progress(f"rank{rank}", worker_progress, f"Sample {rank_sample_start}/{sample_count}"))

            window = lazy_loader.resolve_batch(
                sample_index=rank_sample_start,
                batch_size=requested_batch_size,
                batch_index=rank_sample_start,
            )
        else:
            requested_batch_size = min(adaptive_state.next_batch_size, sample_count - global_sample_cursor)
            if requested_batch_size <= 0:
                global_sample_cursor = 0
                iteration_index += 1
                continue

            worker_progress = float(global_sample_cursor) / sample_count
            print(eng._json_worker_progress(f"rank{rank}", worker_progress, f"Sample {global_sample_cursor}/{sample_count}"))
            window = lazy_loader.resolve_batch(
                sample_index=global_sample_cursor,
                batch_size=requested_batch_size,
                batch_index=global_sample_cursor,
            )

        resolved_batch = window.resolved_batch
        step_samples = _truncate_samples_to_cap(resolved_batch.samples, trajectory_length_cap)
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

        assert data_model_name == expected_model_name, "training data model_official_name must match model_path"
        assert max_input_token_id < model_vocab_size, "input_ids contain token id out of model vocab range"
        assert max_label_token_id < model_vocab_size, "labels contain token id out of model vocab range"

        if torch.cuda.is_available() and device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device=device)
        step_start = time.perf_counter()
        should_sync = (accumulation_step + 1) == config.grad_accum_steps
        sync_context = nullcontext()
        if (
            is_distributed
            and config.training_plan == "lora_current"
            and hasattr(model, "no_sync")
            and not should_sync
        ):
            sync_context = model.no_sync()
        collated = None
        input_ids = None
        labels = None
        attention_mask = None
        advantages = None
        logits = None
        loss_output = None
        try:
            collated = collate_training_samples(samples=step_batch.samples, pad_token_id=pad_token_id)
            input_ids = collated.input_ids.to(device=device, non_blocking=True)
            labels = collated.labels.to(device=device, non_blocking=True)
            attention_mask = collated.attention_mask.to(device=device, non_blocking=True)
            advantages = collated.advantages.to(device=device, non_blocking=True)

            with sync_context:
                logits = eng._forward_logits(model, input_ids=input_ids, attention_mask=attention_mask)
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
            optimizer.zero_grad(set_to_none=True)
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
                batch_token_length=max(sample.input_length for sample in step_batch.samples),
                next_batch_size=(
                    _reduce_trajectory_length_cap(trajectory_length_cap)
                    if adaptive_state.next_batch_size <= 1 and trajectory_length_cap > MIN_TRAJECTORY_LENGTH_CAP
                    else reduced_batch_size
                ),
                will_retry=True,
            )
            if adaptive_state.next_batch_size <= 1:
                if trajectory_length_cap <= MIN_TRAJECTORY_LENGTH_CAP:
                    raise
                next_cap = _reduce_trajectory_length_cap(trajectory_length_cap)
                if is_distributed:
                    if _is_primary_rank():
                        print(
                            "[warning] "
                            "oom_at_batch_size_1=1 "
                            f"trajectory_length_cap_old={trajectory_length_cap} "
                            f"trajectory_length_cap_new={next_cap}"
                        )
                        print(
                            "[status] "
                            f"adaptive_trajectory_cap_reduced=1 old_cap={trajectory_length_cap} "
                            f"new_cap={next_cap} reason=oom_at_batch_size_1"
                        )
                        trajectory_length_cap = next_cap
                    cap_tensor = torch.tensor([trajectory_length_cap], dtype=torch.int64, device=device)
                    torch.distributed.broadcast(cap_tensor, src=0)
                    trajectory_length_cap = int(cap_tensor.item())
                    _emit_trajectory_length_cap(cap=trajectory_length_cap, eng=eng)
                else:
                    if _is_primary_rank():
                        print(
                            "[warning] "
                            "oom_at_batch_size_1=1 "
                            f"trajectory_length_cap_old={trajectory_length_cap} "
                            f"trajectory_length_cap_new={next_cap}"
                        )
                        print(
                            "[status] "
                            f"adaptive_trajectory_cap_reduced=1 old_cap={trajectory_length_cap} "
                            f"new_cap={next_cap} reason=oom_at_batch_size_1"
                        )
                    trajectory_length_cap = next_cap
                    _emit_trajectory_length_cap(cap=trajectory_length_cap, eng=eng)
                eng._release_step_memory(device)
                if torch.cuda.is_available() and device.type == "cuda":
                    torch.cuda.synchronize(device=device)
                continue
            if _is_primary_rank():
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
                adaptive_state = _broadcast_adaptive_state_distributed(adaptive_state=adaptive_state)
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
            continue

        step_elapsed_sec = max(time.perf_counter() - step_start, 1e-6)
        throughput_samples_per_sec = float(len(step_batch.samples)) / step_elapsed_sec
        measured_tokens_per_sample = float(collated.attention_mask.sum(dim=1).float().mean().item())
        gpu_memory_usage_pct = 100.0 * eng._gpu_memory_utilization(device)
        if torch.cuda.is_available() and device.type == "cuda":
            torch.cuda.synchronize(device=device)
        gpu_memory_allocated_pct = 100.0 * eng._gpu_memory_peak_allocated_ratio(device)
        gpu_memory_reserved_pct = 100.0 * eng._gpu_memory_reserved_ratio(device)
        print(eng._json_key_value("throughput_samples_per_sec", f"{throughput_samples_per_sec:.2f}"))
        print(eng._json_key_value("batch_size", str(len(step_batch.samples))))
        print(eng._json_key_value("batch_token_length", str(int(input_ids.shape[1]))))
        print(eng._json_key_value("requested_batch_size", str(requested_batch_size)))
        print(eng._json_key_value("next_batch_size", str(adaptive_state.next_batch_size)))
        print(eng._json_key_value("next_batch_size_int", str(adaptive_state.next_batch_size)))
        print(eng._json_key_value("next_batch_size_float", f"{adaptive_state.next_batch_size_float:.2f}"))
        print(eng._json_key_value("trajectory_length_cap", str(trajectory_length_cap)))
        print(eng._json_key_value("gpu_memory_usage_pct", f"{gpu_memory_usage_pct:.2f}"))
        print(f"gpu_memory_usage_pct: {gpu_memory_usage_pct:.2f}%")
        print(eng._json_key_value("gpu_memory_allocated_pct", f"{gpu_memory_allocated_pct:.2f}"))
        print(f"gpu_memory_allocated_pct: {gpu_memory_allocated_pct:.2f}%")
        print(eng._json_key_value("gpu_memory_reserved_pct", f"{gpu_memory_reserved_pct:.2f}"))
        print(f"gpu_memory_reserved_pct: {gpu_memory_reserved_pct:.2f}%")
        if _is_primary_rank():
            print(eng._json_key_value("global_step", str(global_step)))
            print(eng._json_key_value("iteration", str(iteration_index)))
            print(eng._json_key_value("batch_index", str(step_batch.batch_index)))
            for stat_key, stat_value in loss_output.stats.items():
                print(eng._json_key_value(stat_key, f"{stat_value:.6f}"))

        accumulation_step += 1
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
                    print(
                        "[status] "
                        "adaptive_batch_wrap_reset_applied=1 "
                        f"iteration={iteration_index} next_batch_size={adaptive_state.next_batch_size}"
                    )
                if reset_batch_size_on_wrap:
                    adaptive_state = _broadcast_adaptive_state_distributed(adaptive_state=adaptive_state)
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
                        print(
                            "[status] "
                            "adaptive_batch_wrap_reset_applied=1 "
                            f"iteration={iteration_index} next_batch_size={adaptive_state.next_batch_size}"
                        )

        if accumulation_step == config.grad_accum_steps:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            accumulation_step = 0
            global_step += 1

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
                adaptive_state = _broadcast_adaptive_state_distributed(adaptive_state=adaptive_state)

            now = time.monotonic()
            elapsed_since_last_log_sec = now - clock.last_log_time
            if _is_primary_rank() and elapsed_since_last_log_sec >= config.log_time_interval:
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
                }
                for key, value in loss_output.stats.items():
                    log_payload[key] = value
                eng._log_json_line(logs_path, log_payload)
                clock.last_log_time = now

            elapsed_since_last_checkpoint_sec = now - clock.last_checkpoint_save_time
            if elapsed_since_last_checkpoint_sec >= config.checkpoint_save_time_interval:
                checkpoint_tag = "checkpoints"
                if _is_primary_rank():
                    print(
                        "[status] "
                        f"saving_periodic_checkpoint=1 elapsed_sec={elapsed_since_last_checkpoint_sec:.2f} "
                        f"global_step={global_step} iteration={iteration_index} "
                        f"batch_index={step_batch.batch_index} next_sample_index={global_sample_cursor}"
                    )
                eng._save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    output_dir=checkpoints_parent_dir,
                    checkpoint_tag=checkpoint_tag,
                    training_plan=config.training_plan,
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
                    elapsed_training_time_sec=_elapsed_training_time_sec(clock=clock, now=now),
                )
                clock.last_checkpoint_save_time = now

    accumulation_step, global_step = _flush_partial_gradients(
        optimizer=optimizer,
        accumulation_step=accumulation_step,
        global_step=global_step,
    )

    eng._save_checkpoint(
        model=model,
        optimizer=optimizer,
        output_dir=checkpoints_parent_dir,
        checkpoint_tag="checkpoints",
        training_plan=config.training_plan,
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
        elapsed_training_time_sec=_elapsed_training_time_sec(clock=clock),
    )
    eng._save_final_model_folder(
        model=model,
        training_plan=config.training_plan,
        final_model_output_parent_dir=final_model_output_parent_dir,
        source_model_path=resolved_model_path,
        tokenizer=tokenizer,
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
    resolved_model_path: str,
    tokenizer: Any,
    initial_batch_size: int,
    initial_adaptive_velocity: float,
    target_gpu_memory_utilization: float,
    max_batch_size_cap: int | None,
    reset_batch_size_on_wrap: bool,
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
        resolved_model_path=resolved_model_path,
        tokenizer=tokenizer,
        initial_batch_size=initial_batch_size,
        initial_adaptive_velocity=initial_adaptive_velocity,
        target_gpu_memory_utilization=target_gpu_memory_utilization,
        max_batch_size_cap=max_batch_size_cap,
        reset_batch_size_on_wrap=reset_batch_size_on_wrap,
        eng=eng,
    )
