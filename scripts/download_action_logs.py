from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import _bootstrap  # noqa: F401

from src_py.modal.modal_experiment_paths import (
    action_logs_artifact_name,
    experiment_local_action_logs_dir,
    experiment_service_state_volume_name,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _reset_local_destination(local_destination: Path) -> None:
    if local_destination.is_dir():
        shutil.rmtree(local_destination)
    elif local_destination.exists():
        local_destination.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Download a specific Modal action log folder for a given experiment, epoch, and split"
        )
    )
    parser.add_argument("--model-cli-name", required=True)
    parser.add_argument("--config-nickname", required=True)
    parser.add_argument("--epoch", required=True, type=int)
    parser.add_argument("--split", required=True, choices=["train", "validation"])
    args = parser.parse_args()

    repo_root = _repo_root()
    service_state_volume_name = experiment_service_state_volume_name(
        args.model_cli_name, args.config_nickname
    )
    local_destination = experiment_local_action_logs_dir(
        repo_root,
        args.model_cli_name,
        args.config_nickname,
        args.epoch,
        args.split,
    )
    destination_parent = local_destination.parent
    remote_source = (
        f"medium_files/{args.model_cli_name}/{args.config_nickname}/"
        f"epoch_{args.epoch}/{action_logs_artifact_name(args.split)}"
    )

    _reset_local_destination(local_destination)
    destination_parent.mkdir(parents=True, exist_ok=True)

    command = [
        "uv",
        "run",
        "modal",
        "volume",
        "get",
        "--force",
        service_state_volume_name,
        remote_source,
        str(destination_parent),
    ]
    result = subprocess.run(command, cwd=str(repo_root), check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"modal volume get failed with exit code {result.returncode}"
        )

    if not local_destination.exists():
        raise RuntimeError(
            f"modal volume get completed but the expected action log path is still missing: {local_destination}"
        )

    print(
        (
            f"Downloaded {remote_source} from volume "
            f"'{service_state_volume_name}' to {local_destination}"
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
