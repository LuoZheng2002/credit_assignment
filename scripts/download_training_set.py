from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import _bootstrap  # noqa: F401

from src_py.modal.modal_experiment_paths import (
    experiment_local_training_trajectories_file_path,
    experiment_local_training_trajectories_stats_file_path,
    experiment_remote_training_trajectories_file_path,
    experiment_remote_training_trajectories_stats_file_path,
    experiment_service_state_volume_name,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _reset_local_destination(local_destination: Path) -> None:
    if local_destination.is_dir():
        shutil.rmtree(local_destination)
    elif local_destination.exists():
        local_destination.unlink()


def _download_volume_path(
    repo_root: Path,
    service_state_volume_name: str,
    remote_source: str,
    local_destination: Path,
) -> None:
    _reset_local_destination(local_destination)
    local_destination.parent.mkdir(parents=True, exist_ok=True)

    command = [
        "uv",
        "run",
        "modal",
        "volume",
        "get",
        "--force",
        service_state_volume_name,
        remote_source,
        str(local_destination.parent),
    ]
    result = subprocess.run(command, cwd=str(repo_root), check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"modal volume get failed with exit code {result.returncode}"
        )

    print(
        (
            f"Downloaded {remote_source} from volume "
            f"'{service_state_volume_name}' to {local_destination}"
        ),
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Download the Modal training set artifacts for a specific experiment and epoch"
        )
    )
    parser.add_argument("--model-cli-name", required=True)
    parser.add_argument("--config-nickname", required=True)
    parser.add_argument("--epoch", required=True, type=int)
    args = parser.parse_args()

    repo_root = _repo_root()
    service_state_volume_name = experiment_service_state_volume_name(
        args.model_cli_name, args.config_nickname
    )

    training_trajectories_local = experiment_local_training_trajectories_file_path(
        repo_root, args.model_cli_name, args.config_nickname, args.epoch
    )
    training_trajectories_remote = experiment_remote_training_trajectories_file_path(
        args.model_cli_name, args.config_nickname, args.epoch
    )
    _download_volume_path(
        repo_root,
        service_state_volume_name,
        training_trajectories_remote,
        training_trajectories_local,
    )

    training_trajectories_stats_local = (
        experiment_local_training_trajectories_stats_file_path(
            repo_root, args.model_cli_name, args.config_nickname, args.epoch
        )
    )
    training_trajectories_stats_remote = (
        experiment_remote_training_trajectories_stats_file_path(
            args.model_cli_name, args.config_nickname, args.epoch
        )
    )
    _download_volume_path(
        repo_root,
        service_state_volume_name,
        training_trajectories_stats_remote,
        training_trajectories_stats_local,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
