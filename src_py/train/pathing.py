from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any


def resolve_artifact_root_dir(payload: Mapping[str, Any] | Any) -> Path:
    hpc_training_root_dir = _payload_get(payload, "hpc_training_root_dir")
    if isinstance(hpc_training_root_dir, str) and hpc_training_root_dir.strip():
        return Path(hpc_training_root_dir).expanduser().resolve()
    artifact_root_dir = _payload_get(payload, "artifact_root_dir")
    return Path(str(artifact_root_dir)).expanduser().resolve()


def _payload_get(payload: Mapping[str, Any] | Any, key: str) -> Any:
    if isinstance(payload, Mapping):
        return payload.get(key)
    return getattr(payload, key)


def model_parent_dir(
    artifact_root_dir: Path, model_cli_name: str, config_nickname: str, epoch: int
) -> Path:
    if epoch == 0:
        return artifact_root_dir / "results" / model_cli_name
    return (
        artifact_root_dir
        / "results"
        / model_cli_name
        / config_nickname
        / f"epoch_{epoch}"
    )


def checkpoint_parent_dir(
    artifact_root_dir: Path, model_cli_name: str, config_nickname: str, epoch: int
) -> Path:
    return (
        artifact_root_dir
        / "results"
        / model_cli_name
        / config_nickname
        / f"epoch_{epoch}"
    )


def final_model_output_parent_dir(
    artifact_root_dir: Path,
    model_cli_name: str,
    config_nickname: str,
    epoch: int,
) -> Path:
    return (
        artifact_root_dir
        / "results"
        / model_cli_name
        / config_nickname
        / f"epoch_{epoch + 1}"
    )
