from __future__ import annotations

import argparse
import tomllib
from pathlib import Path
from typing import Any

from .engine import TrainConfig, train
from .pathing import (
    checkpoint_parent_dir,
    final_model_output_parent_dir,
    model_parent_dir,
    resolve_artifact_root_dir,
)
from .status_log_buffer import install_status_log_buffer, shutdown_status_log_buffer


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train causal LM from isolated job folder")
    parser.add_argument("--job-folder-path", type=str, required=True)
    return parser


def _load_train_config_from_job_folder(job_folder_path: str) -> TrainConfig:
    job_folder = Path(job_folder_path)
    assert job_folder.exists(), f"job folder not found: {job_folder}"
    assert job_folder.is_dir(), f"job folder must be a directory: {job_folder}"

    config_path = job_folder / "train_request.toml"
    assert config_path.exists(), f"train request file not found: {config_path}"

    payload: Any
    with config_path.open("rb") as handle:
        payload = tomllib.load(handle)
    assert isinstance(payload, dict), "config toml root must be a table"

    artifact_root_dir = resolve_artifact_root_dir(payload)
    model_cli_name = str(payload["model_cli_name"])
    config_nickname = str(payload["config_nickname"])
    epoch = int(payload["epoch"])
    training_trajectory_sqlite_path = job_folder / "input" / "training_trajectories.sqlite"

    assert training_trajectory_sqlite_path.exists(), (
        "training trajectory sqlite was not uploaded to job folder: "
        f"{training_trajectory_sqlite_path}"
    )

    model_parent_dir_path = model_parent_dir(artifact_root_dir, model_cli_name, config_nickname, epoch)
    checkpoints_parent_dir_path = checkpoint_parent_dir(
        artifact_root_dir, model_cli_name, config_nickname, epoch
    )
    final_model_output_parent_dir_path = final_model_output_parent_dir(
        artifact_root_dir,
        model_cli_name,
        config_nickname,
        epoch,
    )

    training_summary_parent_dir = checkpoints_parent_dir_path

    return TrainConfig(
        training_plan=str(payload["training_plan"]),
        model_parent_dir=str(model_parent_dir_path),
        training_trajectory_sqlite_path=str(training_trajectory_sqlite_path),
        checkpoints_parent_dir=str(checkpoints_parent_dir_path),
        final_model_output_parent_dir=str(final_model_output_parent_dir_path),
        training_summary_parent_dir=str(training_summary_parent_dir),
        advantage_clip=float(payload["advantage_clip"]),
        learning_rate=float(payload["learning_rate"]),
        weight_decay=float(payload["weight_decay"]),
        training_time=float(payload["training_time"]),
        num_iterations_limit=int(payload["num_iterations_limit"]),
        grad_accum_steps=int(payload["grad_accum_steps"]),
        log_time_interval=float(payload["log_time_interval"]),
        checkpoint_save_time_interval=float(payload["checkpoint_save_time_interval"]),
        lora_rank=int(payload.get("lora_rank") or 64),
        lora_alpha=int(payload.get("lora_alpha") or 128),
        lora_dropout=float(payload.get("lora_dropout") or 0.05),
        lora_target_modules_csv=str(payload.get("lora_target_modules_csv") or "q_proj,k_proj,v_proj,o_proj"),
        resume_checkpoint_tag=str(payload.get("resume_checkpoint_tag") or "auto"),
        seed=int(payload["seed"]),
    )


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    install_status_log_buffer(args.job_folder_path)
    config = _load_train_config_from_job_folder(job_folder_path=args.job_folder_path)
    try:
        train(config)
    finally:
        shutdown_status_log_buffer()


if __name__ == "__main__":
    main()
