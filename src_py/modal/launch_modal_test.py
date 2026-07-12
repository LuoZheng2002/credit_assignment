from __future__ import annotations

import fcntl
import importlib
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import modal
from src_py.modal.modal_experiment_paths import experiment_service_state_volume_name

CONFIG_PATH = Path("src_py/modal/test_config.json")
CONFIG_LOCK_PATH = Path("src_py/modal/launcher_config.lock")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


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


def _extract_optional_cli_arg(cli_args: list[str], flag_name: str) -> str | None:
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
    return None


def _test_app_name(model_cli_name: str, config_nickname: str) -> str:
    return f"credit-assignment-test-{model_cli_name}-{config_nickname}"


def _require_dict_value(payload: dict[str, Any], key: str, *, context: str) -> Any:
    if key not in payload:
        raise RuntimeError(f"{context} missing required field '{key}'")
    return payload[key]


def _require_string(payload: dict[str, Any], key: str, *, context: str) -> str:
    value = _require_dict_value(payload, key, context=context)
    if not isinstance(value, str):
        raise RuntimeError(f"{context} field '{key}' must be a string")
    value = value.strip()
    if not value:
        raise RuntimeError(f"{context} field '{key}' must be non-empty")
    return value


def _require_int(payload: dict[str, Any], key: str, *, context: str) -> int:
    value = _require_dict_value(payload, key, context=context)
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(f"{context} field '{key}' must be an integer")
    return int(value)


def _require_bool(payload: dict[str, Any], key: str, *, context: str) -> bool:
    value = _require_dict_value(payload, key, context=context)
    if not isinstance(value, bool):
        raise RuntimeError(f"{context} field '{key}' must be a boolean")
    return value


def _normalize_testing_config_entry(entry: Any, *, context: str) -> dict[str, Any]:
    if not isinstance(entry, dict):
        raise RuntimeError(f"{context} must be an object, got {type(entry)}")
    return {
        "model_cli_name": _require_string(entry, "model_cli_name", context=context),
        "config_nickname": _require_string(entry, "config_nickname", context=context),
        "testing_rollout_config_path": _require_string(
            entry, "testing_rollout_config_path", context=context
        ),
        "epoch": _require_int(entry, "epoch", context=context),
        "total_epochs": _require_int(entry, "total_epochs", context=context),
        "max_rollout_concurrency": _require_int(
            entry, "max_rollout_concurrency", context=context
        ),
        "ui": _require_bool(entry, "ui", context=context),
        "rollout_time_limit_secs": _require_int(
            entry, "rollout_time_limit_secs", context=context
        ),
        "max_python_processes": _require_int(
            entry, "max_python_processes", context=context
        ),
        "num_gpus": _require_int(entry, "num_gpus", context=context),
    }


def _load_testing_configs_from_path(config_path: Path) -> list[dict[str, Any]]:
    try:
        raw = config_path.read_text(encoding="utf-8")
    except OSError as error:
        raise RuntimeError(
            f"failed to read testing configs JSON at {config_path}: {error}"
        ) from error
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"invalid testing configs JSON: {error}") from error
    if not isinstance(payload, list):
        raise RuntimeError("testing configs JSON must be a list")
    if not payload:
        raise RuntimeError("testing configs JSON list must be non-empty")
    normalized = [
        _normalize_testing_config_entry(entry, context=f"testing config entry[{index}]")
        for index, entry in enumerate(payload)
    ]
    reference = normalized[0]
    for index, entry in enumerate(normalized[1:], start=1):
        for key in (
            "max_rollout_concurrency",
            "ui",
            "rollout_time_limit_secs",
            "max_python_processes",
            "num_gpus",
        ):
            if entry[key] != reference[key]:
                raise RuntimeError(
                    f"testing config entry[{index}] field '{key}' must match entry[0]; "
                    f"got {entry[key]!r} vs {reference[key]!r}"
                )
    return normalized


def _load_testing_configs_from_cli(cli_args: list[str]) -> list[dict[str, Any]]:
    source_path_raw = _extract_optional_cli_arg(cli_args, "--testing-configs-path")
    if source_path_raw is not None:
        return _load_testing_configs_from_path(Path(source_path_raw))

    return [
        {
            "model_cli_name": _extract_required_cli_arg(cli_args, "--model-cli-name"),
            "config_nickname": _extract_required_cli_arg(cli_args, "--config-nickname"),
            "testing_rollout_config_path": _extract_required_cli_arg(
                cli_args, "--testing-rollout-config-path"
            ),
            "posterior_hyperparameters_path": _extract_required_cli_arg(
                cli_args, "--posterior-hyperparameters-path"
            ),
            "epoch": int(_extract_required_cli_arg(cli_args, "--epoch")),
            "total_epochs": int(_extract_required_cli_arg(cli_args, "--total-epochs")),
            "max_rollout_concurrency": int(
                _extract_required_cli_arg(cli_args, "--max-rollout-concurrency")
            ),
            "ui": "--ui" in cli_args,
            "rollout_time_limit_secs": int(
                _extract_required_cli_arg(cli_args, "--rollout-time-limit-secs")
            ),
            "max_python_processes": int(
                _extract_optional_cli_arg(cli_args, "--max-python-processes") or "1"
            ),
            "num_gpus": int(_extract_required_cli_arg(cli_args, "--num-gpus")),
        }
    ]


def _repo_scoped_config_path(repo_root: Path) -> Path:
    return repo_root / CONFIG_PATH


def _testing_mount_dir(model_cli_name: str, config_nickname: str) -> str:
    volume_name = experiment_service_state_volume_name(model_cli_name, config_nickname)
    return f"/volume/{volume_name}"


def _add_mount_dir_to_testing_configs(
    testing_configs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    augmented: list[dict[str, Any]] = []
    for entry in testing_configs:
        augmented.append(
            {
                **entry,
                "mount_dir": _testing_mount_dir(
                    entry["model_cli_name"], entry["config_nickname"]
                ),
            }
        )
    return augmented


def _write_test_config(repo_root: Path, testing_configs: list[dict[str, Any]]) -> Path:
    config_path = _repo_scoped_config_path(repo_root)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps(testing_configs, ensure_ascii=True), encoding="utf-8"
    )
    return config_path


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


@contextmanager
def _config_file_lock(repo_root: Path):
    lock_path = repo_root / CONFIG_LOCK_PATH
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a", encoding="utf-8") as lock_file:
        print("Waiting for config file lock...", flush=True)
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _launch_test_app(repo_root: Path):
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

    test_module = importlib.import_module("src_py.modal.modal_test_app")
    app = getattr(test_module, "app")
    service_cls = getattr(test_module, "TestService")

    with modal.enable_output():
        with app.run(detach=True):
            instance = service_cls()
            return instance.run_test.spawn()


def main() -> int:
    cli_args = _strip_legacy_compute_backend(sys.argv[1:])
    repo_root = _repo_root()
    testing_configs = _load_testing_configs_from_cli(cli_args)
    testing_configs = _add_mount_dir_to_testing_configs(testing_configs)
    first_config = testing_configs[0]
    num_gpus = first_config["num_gpus"]
    model_cli_name = first_config["model_cli_name"]
    config_nickname = first_config["config_nickname"]
    mount_dir = first_config["mount_dir"]
    service_state_volume_name = experiment_service_state_volume_name(
        model_cli_name, config_nickname
    )
    app_name = _test_app_name(model_cli_name, config_nickname)
    print(f"Validated test num_gpus: {num_gpus}", flush=True)
    print(
        f"Validated test config batch size: {len(testing_configs)}",
        flush=True,
    )
    print(
        "Validated test runtime: local wrapper-managed inference/training",
        flush=True,
    )
    print(f"Validated test mount dir: {mount_dir}", flush=True)
    print(f"Resolved experiment volume name: {service_state_volume_name}", flush=True)
    print(f"Resolved Modal app name: {app_name}", flush=True)

    test_call = None
    config_path = repo_root / CONFIG_PATH
    with _config_file_lock(repo_root):
        try:
            config_path = _write_test_config(repo_root, testing_configs)
            print(f"Wrote test config: {config_path}", flush=True)
            print(
                f"Submitting Modal test app in detached mode: {app_name}",
                flush=True,
            )
            test_call = _launch_test_app(repo_root)
        finally:
            _remove_config_file(config_path)
            print(f"Removed test config: {config_path}", flush=True)

    if test_call is None:
        raise RuntimeError("failed to submit Modal test call")

    call_id = getattr(test_call, "object_id", None)
    if call_id is not None:
        print(f"Submitted test call id: {call_id}", flush=True)
    else:
        print("Submitted test call", flush=True)

    print(
        "The launcher has exited after successfully submitting the test job; "
        "the remote Modal call will continue independently.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
