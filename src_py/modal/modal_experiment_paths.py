from __future__ import annotations

import hashlib
import re
from pathlib import Path

MAX_MODAL_OBJECT_NAME_LENGTH = 64
LOCAL_RESULTS_DIR = Path("results")
SERVICE_STATE_VOLUME_PREFIX = "credit-assignment"


def sanitize_modal_name_component(value: str) -> str:
    sanitized = re.sub(r"[^a-z0-9-]+", "-", value.lower())
    sanitized = re.sub(r"-+", "-", sanitized).strip("-")
    if not sanitized:
        raise RuntimeError(f"invalid empty Modal name component derived from {value!r}")
    return sanitized


def experiment_service_state_volume_name(
    model_cli_name: str, config_nickname: str, pipeline: str = ""
) -> str:
    """Derive a deterministic Modal volume name.

    When *pipeline* is non-empty it is inserted between the prefix and model
    name components so that different pipelines (e.g. orchestrator, oneshot
    rollout, oneshot training) targeting the same model / config never share
    the same volume.
    """
    model_component = sanitize_modal_name_component(model_cli_name)
    config_component = sanitize_modal_name_component(config_nickname)

    if pipeline:
        pipeline_component = sanitize_modal_name_component(pipeline)
        components = [pipeline_component, model_component, config_component]
        hash_input = f"{pipeline}\0{model_cli_name}\0{config_nickname}"
    else:
        components = [model_component, config_component]
        hash_input = f"{model_cli_name}\0{config_nickname}"

    full_name = f"{SERVICE_STATE_VOLUME_PREFIX}-{'-'.join(components)}"
    if len(full_name) <= MAX_MODAL_OBJECT_NAME_LENGTH:
        return full_name

    digest = hashlib.sha1(hash_input.encode("utf-8")).hexdigest()[:10]
    num_dashes = len(components) + 1
    remaining = (
        MAX_MODAL_OBJECT_NAME_LENGTH
        - len(SERVICE_STATE_VOLUME_PREFIX)
        - len(digest)
        - num_dashes
    )
    if remaining <= 2:
        raise RuntimeError(
            "Modal volume name prefix is too long to derive a per-experiment volume name"
        )

    num_parts = len(components)
    budgets: list[int] = []
    budget_remaining = remaining
    for i in range(num_parts):
        budget = max(1, budget_remaining // (num_parts - i))
        budgets.append(budget)
        budget_remaining -= budget

    truncated_names: list[str] = []
    for comp, budget in zip(components, budgets):
        truncated = comp[:budget].rstrip("-") or comp[:1]
        truncated_names.append(truncated)

    return f"{SERVICE_STATE_VOLUME_PREFIX}-{'-'.join(truncated_names)}-{digest}"


def experiment_remote_small_files_dir(model_cli_name: str, config_nickname: str) -> str:
    return f"small_files/{model_cli_name}/{config_nickname}"


def experiment_local_small_files_dir(
    repo_root: Path, model_cli_name: str, config_nickname: str
) -> Path:
    return (
        repo_root / LOCAL_RESULTS_DIR / "small_files" / model_cli_name / config_nickname
    )


def experiment_remote_large_files_dir(model_cli_name: str, config_nickname: str) -> str:
    return f"large_files/{model_cli_name}/{config_nickname}"


def experiment_local_large_files_dir(
    repo_root: Path, model_cli_name: str, config_nickname: str
) -> Path:
    return (
        repo_root / LOCAL_RESULTS_DIR / "large_files" / model_cli_name / config_nickname
    )


def experiment_local_medium_files_dir(
    repo_root: Path, model_cli_name: str, config_nickname: str
) -> Path:
    return repo_root / LOCAL_RESULTS_DIR / "medium_files"


def training_trajectories_dir_name() -> str:
    return "training_trajectories"


def training_trajectories_msgpack_file_name() -> str:
    return "trajectories.msgpack"


def training_trajectories_stats_file_name() -> str:
    return "training_trajectories_stats.json"


def training_trajectories_config_bundle_file_name() -> str:
    return "config_bundle.json"


def experiment_remote_training_trajectories_file_path(
    model_cli_name: str, config_nickname: str, epoch: int
) -> str:
    return (
        f"medium_files/{model_cli_name}/{config_nickname}/epoch_{epoch}/"
        f"{training_trajectories_dir_name()}"
    )


def experiment_local_training_trajectories_file_path(
    repo_root: Path, model_cli_name: str, config_nickname: str, epoch: int
) -> Path:
    return (
        repo_root
        / LOCAL_RESULTS_DIR
        / "medium_files"
        / model_cli_name
        / config_nickname
        / f"epoch_{epoch}"
        / training_trajectories_dir_name()
    )


def experiment_remote_training_trajectories_stats_file_path(
    model_cli_name: str, config_nickname: str, epoch: int
) -> str:
    return (
        f"medium_files/{model_cli_name}/{config_nickname}/epoch_{epoch}/"
        f"{training_trajectories_stats_file_name()}"
    )


def experiment_local_training_trajectories_stats_file_path(
    repo_root: Path, model_cli_name: str, config_nickname: str, epoch: int
) -> Path:
    return (
        repo_root
        / LOCAL_RESULTS_DIR
        / "medium_files"
        / model_cli_name
        / config_nickname
        / f"epoch_{epoch}"
        / training_trajectories_stats_file_name()
    )


def experiment_remote_training_trajectories_config_bundle_file_path(
    model_cli_name: str, config_nickname: str, epoch: int
) -> str:
    return (
        f"medium_files/{model_cli_name}/{config_nickname}/epoch_{epoch}/"
        f"{training_trajectories_dir_name()}/"
        f"{training_trajectories_config_bundle_file_name()}"
    )


def experiment_local_training_trajectories_config_bundle_file_path(
    repo_root: Path, model_cli_name: str, config_nickname: str, epoch: int
) -> Path:
    return (
        repo_root
        / LOCAL_RESULTS_DIR
        / "medium_files"
        / model_cli_name
        / config_nickname
        / f"epoch_{epoch}"
        / training_trajectories_dir_name()
        / training_trajectories_config_bundle_file_name()
    )


def action_logs_artifact_name(split: str) -> str:
    if split == "train":
        return "action_logs_training.extsort"
    if split == "validation":
        return "action_logs_validation.extsort"
    raise RuntimeError(f"unsupported split {split!r}; expected 'train' or 'validation'")


def experiment_local_action_logs_dir(
    repo_root: Path,
    model_cli_name: str,
    config_nickname: str,
    epoch: int,
    split: str,
) -> Path:
    return (
        experiment_local_medium_files_dir(repo_root, model_cli_name, config_nickname)
        / model_cli_name
        / config_nickname
        / f"epoch_{epoch}"
        / action_logs_artifact_name(split)
    )
