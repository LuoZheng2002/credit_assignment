from __future__ import annotations

from pathlib import Path
from typing import Any


def resolve_artifact_root_dir(payload: dict[str, Any]) -> Path:
    hpc_training_root_dir = payload.get("hpc_training_root_dir")
    if isinstance(hpc_training_root_dir, str) and hpc_training_root_dir.strip():
        return Path(hpc_training_root_dir).expanduser().resolve()
    return Path(str(payload["artifact_root_dir"])).expanduser().resolve()


def model_parent_dir(artifact_root_dir: Path, model_cli_name: str, config_nickname: str, epoch: int) -> Path:
    if epoch == 0:
        return artifact_root_dir / "results" / model_cli_name
    return artifact_root_dir / "results" / model_cli_name / config_nickname / f"epoch_{epoch}"


def checkpoint_parent_dir(
    artifact_root_dir: Path, model_cli_name: str, config_nickname: str, epoch: int
) -> Path:
    return artifact_root_dir / "results" / model_cli_name / config_nickname / f"epoch_{epoch}"


def final_model_output_parent_dir(
    artifact_root_dir: Path,
    model_cli_name: str,
    config_nickname: str,
    epoch: int,
) -> Path:
    return artifact_root_dir / "results" / model_cli_name / config_nickname / f"epoch_{epoch + 1}"
