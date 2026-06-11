from __future__ import annotations

import importlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import modal

APP_NAME = "credit-assignment-orchestrator-service"
CONFIG_PATH = Path("src_py/modal/orchestrator_config.json")
MAX_MODAL_OBJECT_NAME_LENGTH = 64
LOCAL_MODAL_DOWNLOADS_DIR = Path("modal_downloads")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _extract_required_cli_arg(cli_args: list[str], flag_name: str) -> str:
    index = 0
    while index < len(cli_args):
        arg = cli_args[index]
        if arg == flag_name:
            if index + 1 >= len(cli_args):
                raise RuntimeError(f"Missing value after {flag_name}")
            value = cli_args[index + 1].strip()
            if not value:
                raise RuntimeError(f"{flag_name} must be non-empty")
            return value
        prefixed_flag = f"{flag_name}="
        if arg.startswith(prefixed_flag):
            value = arg[len(prefixed_flag) :].strip()
            if not value:
                raise RuntimeError(f"{flag_name} must be non-empty")
            return value
        index += 1
    raise RuntimeError(f"Missing required {flag_name} argument")


def _sanitize_modal_name_component(value: str) -> str:
    sanitized = re.sub(r"[^a-z0-9-]+", "-", value.lower())
    sanitized = re.sub(r"-+", "-", sanitized).strip("-")
    if not sanitized:
        raise RuntimeError(f"invalid empty Modal name component derived from {value!r}")
    return sanitized


def _experiment_service_state_volume_name(
    model_cli_name: str, config_nickname: str
) -> str:
    prefix = "credit-assignment-modal-service-state"
    model_component = _sanitize_modal_name_component(model_cli_name)
    config_component = _sanitize_modal_name_component(config_nickname)
    full_name = f"{prefix}-{model_component}-{config_component}"
    if len(full_name) <= MAX_MODAL_OBJECT_NAME_LENGTH:
        return full_name

    import hashlib

    digest = hashlib.sha1(
        f"{model_cli_name}\0{config_nickname}".encode("utf-8")
    ).hexdigest()[:10]
    remaining = MAX_MODAL_OBJECT_NAME_LENGTH - len(prefix) - len(digest) - 3
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
    return f"{prefix}-{truncated_model}-{truncated_config}-{digest}"


def _experiment_local_small_files_dir(
    repo_root: Path, model_cli_name: str, config_nickname: str
) -> Path:
    return (
        repo_root
        / LOCAL_MODAL_DOWNLOADS_DIR
        / model_cli_name
        / config_nickname
        / "small_files"
    )


def _write_orchestrator_config(
    repo_root: Path, cli_args: list[str], service_state_volume_name: str
) -> Path:
    if not cli_args:
        raise RuntimeError(
            "No orchestrator arguments provided. Pass the same flags you would pass to bin_orchestrator."
        )
    config_path = repo_root / CONFIG_PATH
    config_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "args": cli_args,
        "service_state_volume_name": service_state_volume_name,
    }
    config_path.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")
    return config_path


def _extract_num_gpus(cli_args: list[str]) -> int:
    index = 0
    while index < len(cli_args):
        arg = cli_args[index]
        if arg == "--num-gpus":
            if index + 1 >= len(cli_args):
                raise RuntimeError("Missing value after --num-gpus")
            raw_value = cli_args[index + 1].strip()
            break
        if arg.startswith("--num-gpus="):
            raw_value = arg.split("=", 1)[1].strip()
            break
        index += 1
    else:
        raise RuntimeError(
            "Missing required --num-gpus argument. Ensure orchestrator script passes --num-gpus."
        )

    try:
        num_gpus = int(raw_value)
    except ValueError as error:
        raise RuntimeError(
            f"--num-gpus must be an integer, got: {raw_value!r}"
        ) from error

    if num_gpus <= 0:
        raise RuntimeError(f"--num-gpus must be positive, got: {num_gpus}")
    return num_gpus


def _strip_legacy_compute_backend(cli_args: list[str]) -> list[str]:
    normalized: list[str] = []
    index = 0
    removed = False
    while index < len(cli_args):
        arg = cli_args[index]
        if arg == "--compute-backend":
            if index + 1 >= len(cli_args):
                raise RuntimeError("Missing value after --compute-backend")
            removed = True
            index += 2
            continue
        if arg.startswith("--compute-backend="):
            removed = True
            index += 1
            continue
        normalized.append(arg)
        index += 1
    if removed:
        print(
            "Ignoring legacy --compute-backend flag; orchestration now always uses local wrapper-managed runtime paths.",
            flush=True,
        )
    return normalized


def _remove_config_file(config_path: Path) -> None:
    if config_path.exists():
        config_path.unlink()


def _download_volume_small_files(
    repo_root: Path,
    service_state_volume_name: str,
    model_cli_name: str,
    config_nickname: str,
) -> Path:
    destination = _experiment_local_small_files_dir(
        repo_root, model_cli_name, config_nickname
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        shutil.rmtree(destination)

    command = [
        "modal",
        "volume",
        "get",
        service_state_volume_name,
        "small_files",
        str(destination),
    ]
    result = subprocess.run(
        command,
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip()
        stdout = result.stdout.strip()
        combined = "\n".join(part for part in [stdout, stderr] if part)
        raise RuntimeError(
            "failed to download Modal volume small_files directory "
            f"from volume={service_state_volume_name} to {destination}: {combined}"
        )
    return destination


def _run_orchestration(repo_root: Path) -> dict[str, object]:
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

    orchestrator_module = importlib.import_module("modal_orchestrator_app")
    app = getattr(orchestrator_module, "app")
    service_cls = getattr(orchestrator_module, "OrchestratorService")

    with modal.enable_output():
        with app.run():
            instance = service_cls()
            result = instance.orchestrate.remote()

    if isinstance(result, dict):
        return result
    return {
        "ok": False,
        "error_code": "INVALID_REMOTE_RESULT",
        "error": f"unexpected orchestrate result type: {type(result)}",
    }


def main() -> int:
    cli_args = _strip_legacy_compute_backend(sys.argv[1:])
    num_gpus = _extract_num_gpus(cli_args)
    model_cli_name = _extract_required_cli_arg(cli_args, "--model-cli-name")
    config_nickname = _extract_required_cli_arg(cli_args, "--config-nickname")
    service_state_volume_name = _experiment_service_state_volume_name(
        model_cli_name, config_nickname
    )
    repo_root = _repo_root()
    config_path = _write_orchestrator_config(
        repo_root, cli_args, service_state_volume_name
    )
    local_small_files_dir = _experiment_local_small_files_dir(
        repo_root, model_cli_name, config_nickname
    )
    print(f"Wrote orchestrator config: {config_path}", flush=True)
    print(f"Validated orchestrator num_gpus: {num_gpus}", flush=True)
    print(
        "Validated orchestrator runtime: local wrapper-managed inference/training",
        flush=True,
    )
    print(
        f"Resolved experiment volume name: {service_state_volume_name}",
        flush=True,
    )
    print(
        f"Resolved local small_files download dir: {local_small_files_dir}",
        flush=True,
    )

    result: dict[str, object] | None = None
    orchestration_error: Exception | None = None
    download_error: Exception | None = None
    try:
        print(
            f"Running Modal orchestration app in a single ephemeral step: {APP_NAME}",
            flush=True,
        )
        result = _run_orchestration(repo_root)
    except Exception as error:
        orchestration_error = error
    finally:
        _remove_config_file(config_path)
        print(f"Removed orchestrator config: {config_path}", flush=True)
        try:
            downloaded_dir = _download_volume_small_files(
                repo_root,
                service_state_volume_name,
                model_cli_name,
                config_nickname,
            )
            print(
                f"Downloaded Modal volume small_files to: {downloaded_dir}",
                flush=True,
            )
        except Exception as error:
            download_error = error

    if orchestration_error is not None:
        if download_error is not None:
            print(
                f"Warning: also failed to download Modal small_files: {download_error}",
                flush=True,
            )
        raise orchestration_error

    assert result is not None
    if download_error is not None:
        raise download_error

    print(f"Orchestrator result: {json.dumps(result, ensure_ascii=True)}", flush=True)
    if bool(result.get("ok", False)):
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
