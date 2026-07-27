from __future__ import annotations

import fcntl
import importlib
import json
import sys
from contextlib import contextmanager
from pathlib import Path

import modal
from src_py.modal.modal_experiment_paths import experiment_service_state_volume_name

CONFIG_PATH = Path("src_py/modal/oneshot_training_config.json")
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


def _oneshot_training_app_name(model_cli_name: str, config_nickname: str) -> str:
    return f"credit-assignment-oneshot-training-{model_cli_name}-{config_nickname}"


def _write_oneshot_training_config(
    repo_root: Path,
    cli_args: list[str],
    service_state_volume_name: str,
    app_name: str,
    gpu_name: str,
    modal_time_limit_hrs: float,
) -> Path:
    if not cli_args:
        raise RuntimeError(
            "No oneshot training arguments provided. Pass the same flags you would pass to bin_oneshot_training."
        )
    config_path = repo_root / CONFIG_PATH
    config_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "args": cli_args,
        "service_state_volume_name": service_state_volume_name,
        "app_name": app_name,
        "gpu_name": gpu_name,
        "modal_time_limit_hrs": modal_time_limit_hrs,
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
            "Missing required --num-gpus argument. Ensure oneshot training script passes --num-gpus."
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


def _extract_gpu_name(cli_args: list[str]) -> str:
    return _extract_required_cli_arg(cli_args, "--gpu-name")


def _extract_modal_time_limit_hrs(cli_args: list[str]) -> float:
    index = 0
    while index < len(cli_args):
        arg = cli_args[index]
        if arg == "--modal-time-limit-hrs":
            if index + 1 >= len(cli_args):
                raise RuntimeError("Missing value after --modal-time-limit-hrs")
            raw_value = cli_args[index + 1].strip()
            break
        if arg.startswith("--modal-time-limit-hrs="):
            raw_value = arg.split("=", 1)[1].strip()
            break
        index += 1
    else:
        return 12.0

    try:
        hrs = float(raw_value)
    except ValueError as error:
        raise RuntimeError(
            f"--modal-time-limit-hrs must be a number, got: {raw_value!r}"
        ) from error

    if hrs <= 0:
        raise RuntimeError(f"--modal-time-limit-hrs must be positive, got: {hrs}")
    return hrs


def _strip_modal_only_args(cli_args: list[str]) -> list[str]:
    normalized: list[str] = []
    index = 0
    removed_compute_backend = False
    removed_gpu_name = False
    removed_modal_time_limit_hrs = False
    while index < len(cli_args):
        arg = cli_args[index]
        if arg == "--compute-backend":
            if index + 1 >= len(cli_args):
                raise RuntimeError("Missing value after --compute-backend")
            removed_compute_backend = True
            index += 2
            continue
        if arg.startswith("--compute-backend="):
            removed_compute_backend = True
            index += 1
            continue
        if arg == "--gpu-name":
            if index + 1 >= len(cli_args):
                raise RuntimeError("Missing value after --gpu-name")
            removed_gpu_name = True
            index += 2
            continue
        if arg.startswith("--gpu-name="):
            removed_gpu_name = True
            index += 1
            continue
        if arg == "--modal-time-limit-hrs":
            if index + 1 >= len(cli_args):
                raise RuntimeError("Missing value after --modal-time-limit-hrs")
            removed_modal_time_limit_hrs = True
            index += 2
            continue
        if arg.startswith("--modal-time-limit-hrs="):
            removed_modal_time_limit_hrs = True
            index += 1
            continue
        normalized.append(arg)
        index += 1
    if removed_compute_backend:
        print(
            "Ignoring legacy --compute-backend flag; oneshot training now always uses local wrapper-managed runtime paths.",
            flush=True,
        )
    if removed_gpu_name:
        print(
            "Stripped Modal-only --gpu-name flag before invoking bin_oneshot_training.",
            flush=True,
        )
    if removed_modal_time_limit_hrs:
        print(
            "Stripped Modal-only --modal-time-limit-hrs flag before invoking bin_oneshot_training.",
            flush=True,
        )
    return normalized


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
            "Ignoring legacy --compute-backend flag; oneshot training now always uses local wrapper-managed runtime paths.",
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


def _launch_oneshot_training(repo_root: Path):
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

    training_module = importlib.import_module("src_py.modal.modal_oneshot_training_app")
    app = getattr(training_module, "app")
    service_cls = getattr(training_module, "OneshotTrainingService")

    with modal.enable_output():
        with app.run(detach=True):
            instance = service_cls()
            return instance.run_training.spawn()


def main() -> int:
    raw_cli_args = _strip_legacy_compute_backend(sys.argv[1:])
    num_gpus = _extract_num_gpus(raw_cli_args)
    gpu_name = _extract_gpu_name(raw_cli_args)
    cli_args = _strip_modal_only_args(raw_cli_args)
    modal_time_limit_hrs = _extract_modal_time_limit_hrs(raw_cli_args)
    model_cli_name = _extract_required_cli_arg(cli_args, "--model-cli-name")
    config_nickname_training = _extract_required_cli_arg(
        cli_args, "--config-nickname-training"
    )
    config_nickname_generation = _extract_required_cli_arg(
        cli_args, "--config-nickname-generation"
    )
    mount_dir = _extract_required_cli_arg(cli_args, "--mount-dir")
    service_state_volume_name = experiment_service_state_volume_name(
        model_cli_name, config_nickname_training, pipeline="training"
    )
    app_name = _oneshot_training_app_name(model_cli_name, config_nickname_training)
    repo_root = _repo_root()
    print(f"Validated oneshot training num_gpus: {num_gpus}", flush=True)
    print(f"Validated Modal gpu_name: {gpu_name}", flush=True)
    print(f"Modal time limit: {modal_time_limit_hrs} hrs", flush=True)
    print(
        "Validated oneshot training runtime: local wrapper-managed inference/training",
        flush=True,
    )
    print(f"Validated oneshot training mount dir: {mount_dir}", flush=True)
    print(
        f"Resolved experiment volume name: {service_state_volume_name}",
        flush=True,
    )

    training_call = None
    config_path = repo_root / CONFIG_PATH
    with _config_file_lock(repo_root):
        try:
            config_path = _write_oneshot_training_config(
                repo_root,
                cli_args,
                service_state_volume_name,
                app_name,
                gpu_name,
                modal_time_limit_hrs,
            )
            print(f"Wrote oneshot training config: {config_path}", flush=True)
            print(
                f"Submitting Modal oneshot training app in detached mode: {app_name}",
                flush=True,
            )
            training_call = _launch_oneshot_training(repo_root)
        finally:
            _remove_config_file(config_path)
            print(f"Removed oneshot training config: {config_path}", flush=True)

    if training_call is None:
        raise RuntimeError("failed to submit Modal oneshot training call")

    call_id = getattr(training_call, "object_id", None)
    if call_id is not None:
        print(f"Submitted oneshot training call id: {call_id}", flush=True)
    else:
        print("Submitted oneshot training call", flush=True)

    print(
        "The launcher has exited after successfully submitting the oneshot training job; "
        "the remote Modal call will continue independently.",
        flush=True,
    )
    print(
        "Download the corresponding small_files folder with: "
        "uv run --project pyprojects/minimal python scripts/download_oneshot_training_small_files.py "
        f"--model-cli-name {model_cli_name} --config-nickname-training {config_nickname_training}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
