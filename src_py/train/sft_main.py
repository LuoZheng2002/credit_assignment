from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import msgpack
import torch
from torch.distributed.elastic.multiprocessing.errors import record

from src_py.tui_logging import _tui_info

from .cli_args import (
    SftTrainProcessLaunchArgs,
    TrainingRequestArgs,
    add_model_arguments,
    parse_model_args,
    parse_model_json_file,
)
from .engine import (
    ResumeState,
    _build_fsdp_model,
    _build_full_model,
    _build_lora_model,
    _get_rank_world_size,
    _init_distributed_device,
    _is_primary_rank,
    _make_adam_fp32_aware,
    _resolve_local_model_path,
    _resolve_pad_token_id,
    _resolve_resume_checkpoint_tag,
    _save_checkpoint,
    _save_final_model_folder,
    _set_seed,
)
from .losses import compute_sft_causal_lm_loss
from .status_log_buffer import install_status_log_buffer, shutdown_status_log_buffer
from .training_plan import (
    TRAINING_PLAN_DDP,
    TRAINING_PLAN_LORA,
    assert_supported_training_plan,
)


@dataclass
class _SftRawEntry:
    prompt: str
    response: str


@dataclass
class _SftTokenizedSample:
    input_ids: list[int]
    labels: list[int]


def _load_sft_raw_entries(msgpack_path: str) -> list[_SftRawEntry]:
    entries: list[_SftRawEntry] = []
    with open(msgpack_path, "rb") as file:
        unpacker = msgpack.Unpacker(file, raw=False)
        for payload in unpacker:
            assert isinstance(payload, dict), "SFT raw entry must be a dict"
            entries.append(
                _SftRawEntry(
                    prompt=str(payload["prompt"]),
                    response=str(payload["response"]),
                )
            )
    return entries


def _tokenize_sft_entries(
    entries: list[_SftRawEntry],
    tokenizer: Any,
) -> list[_SftTokenizedSample]:
    """Tokenize raw SFT entries using the model's chat template.

    The prompt is passed as a user message and the response as an assistant
    message.  The model's native chat_template is used to format the full
    conversation; label masking is derived by comparing the full
    conversation tokens with the prompt-only tokens.
    """
    samples: list[_SftTokenizedSample] = []
    for entry in entries:
        # Full conversation: user + assistant
        full_messages = [
            {"role": "user", "content": entry.prompt},
            {"role": "assistant", "content": entry.response},
        ]
        full_text = tokenizer.apply_chat_template(
            full_messages, tokenize=False, add_generation_prompt=False
        )

        # Prompt-only: user message with generation prompt
        prompt_messages = [
            {"role": "user", "content": entry.prompt},
        ]
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )

        # Tokenize both
        all_ids = tokenizer.encode(full_text, add_special_tokens=False)
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)

        prompt_len = len(prompt_ids)
        assert prompt_len <= len(all_ids), "prompt must not be longer than full"
        assert all_ids[:prompt_len] == prompt_ids, (
            "full tokens must start with prompt tokens"
        )

        labels = [-100] * prompt_len + all_ids[prompt_len:]

        samples.append(_SftTokenizedSample(input_ids=list(all_ids), labels=labels))

    return samples


def _collate_sft_batch(
    samples: list[_SftTokenizedSample],
    pad_token_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad and collate a batch of SFT tokenized samples."""
    max_len = max(len(s.input_ids) for s in samples)
    batch_size = len(samples)

    input_ids = torch.full(
        (batch_size, max_len), pad_token_id, dtype=torch.long, device=device
    )
    labels = torch.full((batch_size, max_len), -100, dtype=torch.long, device=device)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)

    for i, sample in enumerate(samples):
        seq_len = len(sample.input_ids)
        input_ids[i, :seq_len] = torch.tensor(sample.input_ids, dtype=torch.long)
        labels[i, :seq_len] = torch.tensor(sample.labels, dtype=torch.long)
        attention_mask[i, :seq_len] = 1

    return input_ids, labels, attention_mask


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SFT training entrypoint")
    add_model_arguments(parser, SftTrainProcessLaunchArgs)
    return parser


def _run_sft_training(
    request: TrainingRequestArgs,
    sft_data_path: str,
) -> None:
    training_plan = assert_supported_training_plan(request.training_plan)
    assert request.learning_rate > 0.0, "learning_rate must be positive"
    assert request.weight_decay >= 0.0, "weight_decay must be non-negative"
    assert request.training_time > 0.0, "training_time must be positive"
    assert request.num_iterations_limit > 0, "num_iterations_limit must be positive"
    assert request.grad_accum_steps > 0, "grad_accum_steps must be positive"
    assert 0.0 < request.adam_beta1 < 1.0, "adam_beta1 must be in (0, 1)"
    assert 0.0 < request.adam_beta2 < 1.0, "adam_beta2 must be in (0, 1)"

    from transformers import AutoTokenizer

    _set_seed(request.seed)
    device = _init_distributed_device()
    rank, world_size = _get_rank_world_size()

    resolved_model_path = _resolve_local_model_path(request.model_parent_dir)
    if _is_primary_rank():
        _tui_info(
            f"start_sft=1 training_plan={training_plan} "
            f"world_size={world_size} training_time={request.training_time:.1f}s "
            f"model_path={resolved_model_path} sft_data_path={sft_data_path}"
        )

    tokenizer = AutoTokenizer.from_pretrained(resolved_model_path)
    pad_token_id = _resolve_pad_token_id(tokenizer.pad_token_id, tokenizer.eos_token_id)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = int(tokenizer.eos_token_id)

    # Load and tokenize all SFT entries (tokenization happens in Python
    # using the model's native chat template — works for any HF model).
    raw_entries = _load_sft_raw_entries(sft_data_path)
    if _is_primary_rank():
        _tui_info(f"sft_raw_entries_loaded={len(raw_entries)}")

    tokenized_samples = _tokenize_sft_entries(raw_entries, tokenizer)
    # Sort by length descending for efficient batching
    tokenized_samples.sort(key=lambda s: len(s.input_ids), reverse=True)
    sample_count = len(tokenized_samples)
    if _is_primary_rank():
        _tui_info(f"sft_samples_tokenized={sample_count}")

    if training_plan == TRAINING_PLAN_LORA:
        model, attention_backend = _build_lora_model(
            model_path=resolved_model_path,
            lora_rank=request.lora_rank or 64,
            lora_alpha=request.lora_alpha or 128,
            lora_dropout=request.lora_dropout or 0.05,
            lora_target_modules_csv=request.lora_target_modules_csv
            or "q_proj,k_proj,v_proj,o_proj",
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

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=request.learning_rate,
        weight_decay=request.weight_decay,
        betas=(request.adam_beta1, request.adam_beta2),
    )
    if request.adam_fp32:
        _make_adam_fp32_aware(optimizer)

    if training_plan in {TRAINING_PLAN_LORA, TRAINING_PLAN_DDP} and world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device.index],
            output_device=device.index,
            find_unused_parameters=False,
        )

    checkpoints_parent_dir = Path(request.checkpoints_parent_dir)
    checkpoints_parent_dir.mkdir(parents=True, exist_ok=True)
    final_model_output_parent_dir = Path(request.final_model_output_parent_dir)
    logs_path = checkpoints_parent_dir / "sft_metrics.jsonl"

    expected_model_name = resolved_model_path
    tokenizer_name = tokenizer.name_or_path.strip()
    assert len(expected_model_name) > 0, "model_path cannot be empty"
    assert len(tokenizer_name) > 0, "tokenizer_name_or_path cannot be empty"
    assert tokenizer_name == expected_model_name, (
        "tokenizer name_or_path must exactly match model_path"
    )

    resolved_resume_tag = _resolve_resume_checkpoint_tag(
        output_dir=checkpoints_parent_dir,
        resume_checkpoint_tag=request.resume_checkpoint_tag or "auto",
    )

    # Load checkpoint if available
    resume_state = ResumeState(
        global_step=0,
        next_iteration_index=0,
        next_batch_cursor=0,
        accumulation_step=0,
        next_sample_index=0,
        next_batch_size=1,
    )
    if len(resolved_resume_tag) > 0:
        if _is_primary_rank():
            _tui_info(
                f"loading_resume_checkpoint=1 checkpoint_tag={resolved_resume_tag}"
            )
        from .engine import _load_checkpoint

        resume_state = _load_checkpoint(
            model=model,
            optimizer=optimizer,
            output_dir=checkpoints_parent_dir,
            checkpoint_tag=resolved_resume_tag,
            training_plan=training_plan,
            adam_fp32=request.adam_fp32,
        )
    elif _is_primary_rank():
        _tui_info("loading_resume_checkpoint=0 starting_fresh=1")

    batch_size = 4
    grad_accum_steps = request.grad_accum_steps
    model.train()

    training_start_time = time.time() - resume_state.elapsed_training_time_sec
    sample_index = resume_state.next_sample_index
    global_step = resume_state.global_step
    accumulation_step = resume_state.accumulation_step
    trained_samples = resume_state.samples_trained
    last_log_time = training_start_time
    last_checkpoint_time = training_start_time

    try:
        while True:
            current_elapsed = time.time() - training_start_time
            if current_elapsed >= request.training_time:
                break
            if global_step >= request.num_iterations_limit:
                break

            if sample_index >= sample_count:
                sample_index = 0  # wrap around

            # Grab the next batch of tokenized samples
            batch_end = min(sample_index + batch_size, sample_count)
            batch_samples = tokenized_samples[sample_index:batch_end]
            actual_batch_size = len(batch_samples)

            # If the batch is smaller than batch_size (wrapping), pad with
            # samples from the beginning
            if actual_batch_size < batch_size:
                wrap_count = batch_size - actual_batch_size
                batch_samples = batch_samples + tokenized_samples[:wrap_count]
                sample_index = wrap_count
            else:
                sample_index = batch_end

            input_ids, labels, attention_mask = _collate_sft_batch(
                batch_samples, pad_token_id, device
            )

            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).logits

            loss_output = compute_sft_causal_lm_loss(logits, labels)
            loss = loss_output.loss / grad_accum_steps
            loss.backward()

            accumulation_step += 1

            if accumulation_step >= grad_accum_steps:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    max_norm=1.0,
                )
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                trained_samples += batch_size * grad_accum_steps
                accumulation_step = 0

                # Logging
                if _is_primary_rank():
                    now = time.time()
                    if now - last_log_time >= request.log_time_interval:
                        last_log_time = now
                        elapsed = now - training_start_time
                        _tui_info(
                            f"sft_step={global_step} "
                            f"loss={loss_output.stats['loss_ce']:.6f} "
                            f"samples={trained_samples} "
                            f"elapsed={elapsed:.1f}s"
                        )
                        with logs_path.open("a", encoding="utf-8") as log_file:
                            log_file.write(
                                json.dumps(
                                    {
                                        "step": global_step,
                                        "elapsed_sec": round(elapsed, 3),
                                        **{
                                            k: round(v, 6)
                                            for k, v in loss_output.stats.items()
                                        },
                                    }
                                )
                                + "\n"
                            )

                # Checkpointing
                now = time.time()
                if now - last_checkpoint_time >= request.checkpoint_save_time_interval:
                    last_checkpoint_time = now
                    # Ensure no partial accumulation when checkpointing
                    if accumulation_step != 0:
                        torch.nn.utils.clip_grad_norm_(
                            [p for p in model.parameters() if p.requires_grad],
                            max_norm=1.0,
                        )
                        optimizer.step()
                        optimizer.zero_grad()
                        global_step += 1
                        trained_samples += batch_size * accumulation_step
                        accumulation_step = 0
                    if _is_primary_rank():
                        _tui_info(
                            f"saving_sft_checkpoint=1 global_step={global_step} sample_index={sample_index}"
                        )
                    _save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        output_dir=checkpoints_parent_dir,
                        checkpoint_tag="latest",
                        training_plan=training_plan,
                        global_step=global_step,
                        next_iteration_index=global_step,
                        next_batch_cursor=sample_index,
                        accumulation_step=accumulation_step,
                        next_sample_index=sample_index,
                        next_batch_size=batch_size,
                        adaptive_velocity=0.0,
                        adaptive_throughput_ema=0.0,
                        adaptive_best_throughput_ema=0.0,
                        adaptive_memory_utilization_ema=0.0,
                        adaptive_previous_tokens_per_sample=0.0,
                        adaptive_next_batch_size_float=float(batch_size),
                        elapsed_training_time_sec=time.time() - training_start_time,
                        samples_trained=trained_samples,
                        samples_available=sample_count,
                        max_average_absolute_advantage=-1.0,
                        min_average_absolute_advantage=-1.0,
                        median_average_absolute_advantage=-1.0,
                    )
    finally:
        pass  # No lazy loader to close — all data is in memory

    _save_final_model_folder(
        model=model,
        training_plan=training_plan,
        final_model_output_parent_dir=final_model_output_parent_dir,
        source_model_path=resolved_model_path,
        tokenizer=tokenizer,
    )
    if _is_primary_rank():
        total_elapsed = time.time() - training_start_time
        _tui_info(
            f"sft_finished=1 "
            f"total_steps={global_step} "
            f"total_samples={trained_samples} "
            f"elapsed={total_elapsed:.1f}s "
            f"final_model_dir={final_model_output_parent_dir}"
        )


@record
def main() -> None:
    launch_args = parse_model_args(_build_parser(), SftTrainProcessLaunchArgs)
    install_status_log_buffer(launch_args.orchestrator_socket_path)
    request = parse_model_json_file(
        TrainingRequestArgs, launch_args.training_request_json_path
    )
    try:
        _run_sft_training(request, launch_args.sft_training_data_path)
    finally:
        shutdown_status_log_buffer()


if __name__ == "__main__":
    main()
