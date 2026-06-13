from __future__ import annotations

import fcntl
import importlib
import json
import sys
from contextlib import contextmanager
from pathlib import Path

import modal
from src_py.modal.modal_experiment_paths import experiment_service_state_volume_name

CONFIG_PATH = Path("src_py/modal/test_config.json")
CONFIG_LOCK_PATH = Path("src_py/modal/test_config.lock")


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


def _test_app_name(model_cli_name: str, config_nickname: str) -> str:
    return f"credit-assignment-test-{model_cli_name}-{config_nickname}"


def _write_test_config(
    repo_root: Path,
    cli_args: list[str],
    service_state_volume_name: str,
    app_name: str,
) -> Path:
    if not cli_args:
        raise RuntimeError(
            "No test arguments provided. Pass the same flags you would pass to bin_run_test."
        )
    config_path = repo_root / CONFIG_PATH
    config_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "args": cli_args,
        "service_state_volume_name": service_state_volume_name,
        "app_name": app_name,
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
            "Missing required --num-gpus argument. Ensure test launcher passes --num-gpus."
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
    num_gpus = _extract_num_gpus(cli_args)
    model_cli_name = _extract_required_cli_arg(cli_args, "--model-cli-name")
    config_nickname = _extract_required_cli_arg(cli_args, "--config-nickname")
    storage_medium_files_dir = _extract_required_cli_arg(
        cli_args, "--storage-medium-files-dir"
    )
    service_state_volume_name = experiment_service_state_volume_name(
        model_cli_name, config_nickname
    )
    app_name = _test_app_name(model_cli_name, config_nickname)
    repo_root = _repo_root()
    print(f"Validated test num_gpus: {num_gpus}", flush=True)
    print(
        "Validated test runtime: local wrapper-managed inference/training",
        flush=True,
    )
    print(
        f"Validated test storage medium dir: {storage_medium_files_dir}",
        flush=True,
    )
    print(
        f"Resolved experiment volume name: {service_state_volume_name}",
        flush=True,
    )

    test_call = None
    config_path = repo_root / CONFIG_PATH
    with _config_file_lock(repo_root):
        try:
            config_path = _write_test_config(
                repo_root, cli_args, service_state_volume_name, app_name
            )
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
