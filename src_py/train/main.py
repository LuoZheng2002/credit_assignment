from __future__ import annotations

import argparse
from pathlib import Path

from torch.distributed.elastic.multiprocessing.errors import record

from .cli_args import (
    TrainingRequestArgs,
    TrainProcessLaunchArgs,
    TrainingModeOneShot,
    add_model_arguments,
    parse_model_args,
    parse_model_json_file,
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
    training_trajectory_path = Path(launch_args.training_trajectory_path)
    assert training_trajectory_path.exists(), (
        f"training trajectory file not found: {training_trajectory_path}"
    )

    hp = request.hyperparameters

    if isinstance(request.training_mode, TrainingModeOneShot):
        training_mode = "oneshot"
        training_time = request.training_mode.per_epoch_training_time
        model_parent_dir = request.training_mode.base_model_parent_dir
        training_summary_parent_dir = request.training_mode.training_summary_dir
        final_model_output_parent_dir = request.training_mode.training_summary_dir
        oneshot_num_epochs = request.training_mode.num_oneshot_epochs
        oneshot_model_output_root = request.training_mode.model_output_root
    else:
        training_mode = "orchestration"
        training_time = request.training_mode.training_time
        model_parent_dir = request.training_mode.input_model_parent_dir
        training_summary_parent_dir = request.training_mode.training_summary_dir
        final_model_output_parent_dir = request.training_mode.output_model_parent_dir
        oneshot_num_epochs = 0
        oneshot_model_output_root = ""

    model_parent_dir_path = Path(model_parent_dir)
    training_summary_parent_dir_path = Path(training_summary_parent_dir)
    final_model_output_parent_dir_path = Path(final_model_output_parent_dir)

    return TrainConfig(
        lora_or_full=hp.lora_or_full,
        distributed_strategy=hp.distributed_strategy,
        model_parent_dir=str(model_parent_dir_path),
        training_trajectory_path=str(training_trajectory_path),
        training_trajectory_len_cutoff=request.training_trajectory_len_cutoff,
        training_summary_parent_dir=str(training_summary_parent_dir_path),
        final_model_output_parent_dir=str(final_model_output_parent_dir_path),
        advantage_clip=hp.advantage_clip,
        learning_rate=hp.learning_rate,
        weight_decay=hp.weight_decay,
        training_time=training_time,
        num_iterations_limit=request.num_iterations_limit,
        grad_accum_steps=hp.grad_accum_steps,
        log_time_interval=hp.log_time_interval,
        lora_rank=hp.lora_rank or 64,
        lora_alpha=hp.lora_alpha or 128,
        lora_dropout=hp.lora_dropout or 0.05,
        seed=hp.seed,
        adam_beta1=hp.adam_beta1,
        adam_beta2=hp.adam_beta2,
        lr_warmup_steps=hp.lr_warmup_steps,
        training_mode=training_mode,
        oneshot_num_epochs=oneshot_num_epochs,
        oneshot_model_output_root=oneshot_model_output_root,
    )


@record
def main() -> None:
    launch_args = parse_model_args(_build_parser(), TrainProcessLaunchArgs)
    install_status_log_buffer(launch_args.orchestrator_socket_path)
    request = parse_model_json_file(
        TrainingRequestArgs, launch_args.training_request_json_path
    )
    config = _load_train_config(launch_args, request)
    try:
        train(config)
    finally:
        shutdown_status_log_buffer()


if __name__ == "__main__":
    main()
