from __future__ import annotations

import hashlib
import re
from pathlib import Path

MAX_MODAL_OBJECT_NAME_LENGTH = 64
LOCAL_MODAL_DOWNLOADS_DIR = Path("modal_downloads")
SERVICE_STATE_VOLUME_PREFIX = "credit-assignment"


def sanitize_modal_name_component(value: str) -> str:
    sanitized = re.sub(r"[^a-z0-9-]+", "-", value.lower())
    sanitized = re.sub(r"-+", "-", sanitized).strip("-")
    if not sanitized:
        raise RuntimeError(f"invalid empty Modal name component derived from {value!r}")
    return sanitized


def experiment_service_state_volume_name(
    model_cli_name: str, config_nickname: str
) -> str:
    model_component = sanitize_modal_name_component(model_cli_name)
    config_component = sanitize_modal_name_component(config_nickname)
    full_name = f"{SERVICE_STATE_VOLUME_PREFIX}-{model_component}-{config_component}"
    if len(full_name) <= MAX_MODAL_OBJECT_NAME_LENGTH:
        return full_name

    digest = hashlib.sha1(
        f"{model_cli_name}\0{config_nickname}".encode("utf-8")
    ).hexdigest()[:10]
    remaining = (
        MAX_MODAL_OBJECT_NAME_LENGTH
        - len(SERVICE_STATE_VOLUME_PREFIX)
        - len(digest)
        - 3
    )
    if remaining <= 2:
        raise RuntimeError(
            "Modal volume name prefix is too long to derive a per-experiment volume name"
        )
    model_budget = max(1, remaining // 2)
    config_budget = max(1, remaining - model_budget)
    truncated_model = model_component[:model_budget].rstrip("-") or model_component[:1]
    truncated_config = (
        config_component[:config_budget].rstrip("-") or config_component[:1]
    )
    return (
        f"{SERVICE_STATE_VOLUME_PREFIX}-{truncated_model}-{truncated_config}-{digest}"
    )


def experiment_local_small_files_dir(
    repo_root: Path, model_cli_name: str, config_nickname: str
) -> Path:
    return (
        repo_root
        / LOCAL_MODAL_DOWNLOADS_DIR
        / model_cli_name
        / config_nickname
        / "small_files"
    )


def experiment_local_large_files_dir(
    repo_root: Path, model_cli_name: str, config_nickname: str
) -> Path:
    return (
        repo_root
        / LOCAL_MODAL_DOWNLOADS_DIR
        / model_cli_name
        / config_nickname
        / "large_files"
    )


def experiment_local_medium_files_dir(
    repo_root: Path, model_cli_name: str, config_nickname: str
) -> Path:
    return (
        repo_root
        / LOCAL_MODAL_DOWNLOADS_DIR
        / model_cli_name
        / config_nickname
        / "medium_files"
    )
