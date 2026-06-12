from __future__ import annotations

import argparse
from pathlib import Path

from torch.distributed.elastic.multiprocessing.errors import record

from .cli_args import (
    TrainingRequestArgs,
    TrainProcessLaunchArgs,
    add_model_arguments,
    parse_model_args,
    parse_model_stdin,
)
from .engine import TrainConfig, train
from .status_log_buffer import install_status_log_buffer, shutdown_status_log_buffer


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Primary training entrypoint for isolated job folders"
    )
    add_model_arguments(parser, TrainProcessLaunchArgs)
    return parser


def _load_train_config(
    launch_args: TrainProcessLaunchArgs, request: TrainingRequestArgs
) -> TrainConfig:
    training_trajectory_sqlite_path = Path(launch_args.training_trajectory_sqlite_path)
    assert training_trajectory_sqlite_path.exists(), (
        f"training trajectory sqlite not found: {training_trajectory_sqlite_path}"
    )

    model_parent_dir_path = Path(request.model_parent_dir)
    checkpoints_parent_dir_path = Path(request.checkpoints_parent_dir)
    final_model_output_parent_dir_path = Path(request.final_model_output_parent_dir)

    training_summary_parent_dir = checkpoints_parent_dir_path

    return TrainConfig(
        training_plan=request.training_plan,
        model_parent_dir=str(model_parent_dir_path),
        training_trajectory_sqlite_path=str(training_trajectory_sqlite_path),
        checkpoints_parent_dir=str(checkpoints_parent_dir_path),
        final_model_output_parent_dir=str(final_model_output_parent_dir_path),
        training_summary_parent_dir=str(training_summary_parent_dir),
        advantage_clip=request.advantage_clip,
        learning_rate=request.learning_rate,
        weight_decay=request.weight_decay,
        training_time=request.training_time,
        num_iterations_limit=request.num_iterations_limit,
        grad_accum_steps=request.grad_accum_steps,
        log_time_interval=request.log_time_interval,
        checkpoint_save_time_interval=request.checkpoint_save_time_interval,
        lora_rank=request.lora_rank or 64,
        lora_alpha=request.lora_alpha or 128,
        lora_dropout=request.lora_dropout or 0.05,
        lora_target_modules_csv=request.lora_target_modules_csv
        or "q_proj,k_proj,v_proj,o_proj",
        resume_checkpoint_tag=request.resume_checkpoint_tag or "auto",
        seed=request.seed,
    )


@record
def main() -> None:
    launch_args = parse_model_args(_build_parser(), TrainProcessLaunchArgs)
    install_status_log_buffer(launch_args.orchestrator_socket_path)
    request = parse_model_stdin(TrainingRequestArgs)
    config = _load_train_config(launch_args, request)
    try:
        train(config)
    finally:
        shutdown_status_log_buffer()


if __name__ == "__main__":
    main()
