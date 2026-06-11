from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any

DEFAULT_ARTIFACT_ROOT_DIR = "/mnt/service-state"


def slug(value: str) -> str:
    text = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return text or "x"


def combined_config_name(
    model_cli_name: str,
    model_api_name: str,
    config_nickname: str,
    epoch: int,
) -> str:
    return (
        f"{slug(model_cli_name)}_"
        f"{slug(config_nickname)}_"
        f"e{epoch}_"
        f"{slug(model_api_name)}"
    )


def deployment_name(
    model_cli_name: str,
    model_api_name: str,
    config_nickname: str,
    epoch: int,
) -> str:
    base = f"modal_training_{combined_config_name(model_cli_name, model_api_name, config_nickname, epoch)}"
    return base[:63]


def _deploy_config_path() -> Path:
    return Path(__file__).with_name("training_deploy_config.json")


def _write_deploy_config(
    model_cli_name: str,
    model_api_name: str,
    config_nickname: str,
    epoch: int,
    num_gpus: int,
) -> None:
    if num_gpus <= 0:
        raise RuntimeError(f"num_gpus must be positive, got: {num_gpus}")
    payload = {
        "DEPLOY_MODEL_CLI_NAME": model_cli_name,
        "DEPLOY_MODEL_API_NAME": model_api_name,
        "DEPLOY_CONFIG_NICKNAME": config_nickname,
        "DEPLOY_EPOCH": epoch,
        "DEPLOY_NUM_GPUS": num_gpus,
        "DEPLOY_ARTIFACT_ROOT_DIR": DEFAULT_ARTIFACT_ROOT_DIR,
    }
    _deploy_config_path().write_text(
        json.dumps(payload, ensure_ascii=True), encoding="utf-8"
    )


def _remove_deploy_config() -> None:
    path = _deploy_config_path()
    try:
        if path.exists():
            path.unlink()
    except OSError:
        pass


def load_materialized_deploy_config() -> dict[str, Any]:
    try:
        raw = _deploy_config_path().read_text(encoding="utf-8")
        parsed = json.loads(raw)
    except FileNotFoundError as error:
        raise RuntimeError(
            "missing src_py/modal/training_deploy_config.json; deploy via wrapper-managed deploy path"
        ) from error
    except json.JSONDecodeError as error:
        raise RuntimeError(f"invalid deploy config JSON: {error}") from error
    if not isinstance(parsed, dict):
        raise RuntimeError("invalid deploy config format: expected object")
    result = dict(parsed)
    for key, value in result.items():
        if value is None:
            raise RuntimeError(
                f"materialized deploy config missing required field: {key}"
            )
    required_keys = (
        "DEPLOY_MODEL_CLI_NAME",
        "DEPLOY_MODEL_API_NAME",
        "DEPLOY_CONFIG_NICKNAME",
        "DEPLOY_EPOCH",
        "DEPLOY_NUM_GPUS",
        "DEPLOY_ARTIFACT_ROOT_DIR",
    )
    for key in required_keys:
        if key not in result:
            raise RuntimeError(
                f"materialized deploy config missing required field: {key}"
            )
    return result


def app_is_active(apps: list[dict[str, Any]], app_name: str) -> bool:
    for app in apps:
        if (
            app.get("Description") == app_name
            and str(app.get("State", "")).lower() != "stopped"
        ):
            return True
    return False


def ensure_deployed(
    model_cli_name: str,
    model_api_name: str,
    config_nickname: str,
    epoch: int,
    num_gpus: int,
) -> tuple[str, int]:
    app_name = deployment_name(model_cli_name, model_api_name, config_nickname, epoch)
    _write_deploy_config(
        model_cli_name, model_api_name, config_nickname, epoch, num_gpus
    )
    try:
        result = subprocess.run(
            [
                "uv",
                "run",
                "modal",
                "deploy",
                "modal_training_app.py",
                "--name",
                app_name,
                "--detach",
            ],
            check=False,
        )
        return app_name, result.returncode
    finally:
        _remove_deploy_config()


def ensure_undeployed(
    model_cli_name: str,
    model_api_name: str,
    config_nickname: str,
    epoch: int,
) -> tuple[str, int]:
    app_name = deployment_name(model_cli_name, model_api_name, config_nickname, epoch)
    list_result = subprocess.run(
        ["uv", "run", "modal", "app", "list", "--json"],
        check=False,
        capture_output=True,
        text=True,
    )
    if list_result.returncode != 0:
        return app_name, list_result.returncode
    try:
        apps = json.loads(list_result.stdout)
    except json.JSONDecodeError:
        return app_name, 1
    if not app_is_active(apps, app_name):
        return app_name, 0
    stop_result = subprocess.run(
        ["uv", "run", "modal", "app", "stop", app_name, "--yes"],
        check=False,
    )
    return app_name, stop_result.returncode
