from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import _bootstrap  # noqa: F401

from src_py.modal.modal_experiment_paths import (
    experiment_local_medium_files_dir,
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
        description="Download the Modal medium_files folder for a specific experiment"
    )
    parser.add_argument("--model-cli-name", required=True)
    parser.add_argument("--config-nickname", required=True)
    args = parser.parse_args()

    repo_root = _repo_root()
    service_state_volume_name = experiment_service_state_volume_name(
        args.model_cli_name, args.config_nickname
    )
    local_destination = experiment_local_medium_files_dir(
        repo_root, args.model_cli_name, args.config_nickname
    )
    destination_parent = local_destination.parent

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
        "medium_files",
        str(destination_parent),
    ]
    result = subprocess.run(command, cwd=str(repo_root), check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"modal volume get failed with exit code {result.returncode}"
        )

    print(
        f"Downloaded medium_files from volume '{service_state_volume_name}' to {local_destination}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
