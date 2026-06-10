from __future__ import annotations

import argparse
import ctypes
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from src_py.load_model_to_path import ensure_model_snapshot
from src_py.train.cli_args import (
    TrainingRequestArgs,
    TrainingWrapperLaunchArgs,
    TrainProcessLaunchArgs,
    add_model_arguments,
    model_to_cli_args,
    model_to_json_bytes,
    parse_model_args,
    parse_model_stdin,
)
from src_py.train.pathing import (
    checkpoint_parent_dir,
    final_model_output_parent_dir,
    model_parent_dir,
    resolve_artifact_root_dir,
)
from src_py.tui_logging import (
    _tui_error,
    _tui_info,
    _tui_state,
    configure_tui_forwarder,
)


def _set_process_name(name: str) -> None:
    try:
        libc = ctypes.CDLL(None)
        pr_set_name = 15
        libc.prctl(pr_set_name, name.encode("utf-8")[:15], 0, 0, 0)
    except Exception:
        pass


def _emit_event(payload: dict[str, Any]) -> None:
    line = json.dumps(payload, ensure_ascii=True)
    print(line, flush=True)


def _emit_status(backend: str, status: str, message: str) -> None:
    _emit_event(
        {
            "type": "status",
            "backend": backend,
            "status": status,
            "message": message,
            "timestamp": time.time(),
        }
    )


def _emit_status_with_metrics(
    backend: str,
    status: str,
    message: str,
    metrics: dict[str, Any],
) -> None:
    _emit_event(
        {
            "type": "status",
            "backend": backend,
            "status": status,
            "message": message,
            "metrics": metrics,
            "timestamp": time.time(),
        }
    )


def _emit_result_ok(backend: str, message: str, duration_secs: float) -> None:
    _emit_event(
        {
            "type": "result",
            "backend": backend,
            "ok": True,
            "message": message,
            "duration_secs": duration_secs,
            "timestamp": time.time(),
        }
    )


def _emit_result_error(
    backend: str,
    error_code: str,
    error_message: str,
    duration_secs: float,
) -> None:
    _emit_event(
        {
            "type": "result",
            "backend": backend,
            "ok": False,
            "error_code": error_code,
            "error_message": error_message,
            "duration_secs": duration_secs,
            "timestamp": time.time(),
        }
    )


_TERMINATION_REQUESTED = threading.Event()
_ACTIVE_PROCESS_LOCK = threading.Lock()
_ACTIVE_PROCESS: subprocess.Popen[Any] | None = None
_TRAINING_WRAPPER_LOG_PATH: Path | None = None
_WRAPPER_LOG_FILE_HANDLE: Any | None = None


def _configure_wrapper_log_path(raw_path: str) -> None:
    global _TRAINING_WRAPPER_LOG_PATH, _WRAPPER_LOG_FILE_HANDLE
    path = raw_path.strip()
    if not path:
        raise ValueError("--wrapper-log-path must be non-empty")
    resolved = Path(path).expanduser().resolve()
    if resolved.parent and not resolved.parent.exists():
        resolved.parent.mkdir(parents=True, exist_ok=True)
    _TRAINING_WRAPPER_LOG_PATH = resolved
    _WRAPPER_LOG_FILE_HANDLE = open(resolved, "a", buffering=1, encoding="utf-8")
    os.dup2(_WRAPPER_LOG_FILE_HANDLE.fileno(), 1)
    os.dup2(_WRAPPER_LOG_FILE_HANDLE.fileno(), 2)
    sys.stdout = os.fdopen(1, "w", buffering=1, encoding="utf-8", closefd=False)
    sys.stderr = os.fdopen(2, "w", buffering=1, encoding="utf-8", closefd=False)


def _emit_training_tui_identity(
    training_config: TrainingRequestArgs, hf_model_name: str, trajectory_path: Path
) -> None:
    _tui_info(
        "Training wrapper config: "
        f"model_cli_name={training_config.model_cli_name} "
        f"config_nickname={training_config.config_nickname} "
        f"epoch={training_config.epoch} "
        f"hf_model_name={hf_model_name} "
        f"trajectory_sqlite_path={trajectory_path}"
    )


def _set_active_process(process: subprocess.Popen[Any] | None) -> None:
    global _ACTIVE_PROCESS
    with _ACTIVE_PROCESS_LOCK:
        _ACTIVE_PROCESS = process


def _on_signal(signum: int, _: Any) -> None:
    _TERMINATION_REQUESTED.set()
    _tui_error(f"Training wrapper received signal {signum}")
    _emit_status("wrapper", "cancelling", f"received signal {signum}")
    with _ACTIVE_PROCESS_LOCK:
        process = _ACTIVE_PROCESS
    if process is not None and process.poll() is None:
        process.terminate()


def _install_signal_handlers() -> None:
    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)


def _process_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _start_parent_watchdog(backend: str, poll_secs: float = 2.0) -> threading.Thread:
    parent_pid = os.getppid()

    def _watch() -> None:
        while not _TERMINATION_REQUESTED.is_set():
            current_parent = os.getppid()
            if current_parent != parent_pid or not _process_exists(parent_pid):
                _emit_status(
                    backend,
                    "cancelling",
                    (
                        "parent process exited "
                        f"(initial_ppid={parent_pid}, current_ppid={current_parent}); cancelling training wrapper"
                    ),
                )
                _TERMINATION_REQUESTED.set()
                with _ACTIVE_PROCESS_LOCK:
                    process = _ACTIVE_PROCESS
                if process is not None and process.poll() is None:
                    process.terminate()
                return
            _TERMINATION_REQUESTED.wait(timeout=poll_secs)

    watchdog = threading.Thread(target=_watch, daemon=True)
    watchdog.start()
    return watchdog


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Training wrapper that always launches local training"
    )
    add_model_arguments(parser, TrainingWrapperLaunchArgs)
    return parser


def _ensure_initial_model_if_missing(
    backend: str,
    training_config: TrainingRequestArgs,
    hf_model_name: str,
) -> None:
    epoch = training_config.epoch
    if epoch != 0:
        return
    artifact_root_dir = resolve_artifact_root_dir(training_config)
    model_cli_name = training_config.model_cli_name
    config_nickname = training_config.config_nickname
    parent_dir = model_parent_dir(
        artifact_root_dir, model_cli_name, config_nickname, epoch
    )
    model_dir = parent_dir / "model"
    if model_dir.exists():
        return
    _emit_status(
        backend,
        "starting",
        f"initial model missing at {model_dir}; downloading {hf_model_name}",
    )
    ensure_model_snapshot(parent_dir, hf_model_name)


def _write_subprocess_stdin(process: subprocess.Popen[Any], payload: bytes) -> None:
    if process.stdin is None:
        raise RuntimeError("subprocess stdin was not piped")
    process.stdin.write(payload)
    process.stdin.close()


def _run_hpc_training(
    launch_args: TrainingWrapperLaunchArgs,
    training_config: TrainingRequestArgs,
) -> int:
    started_at = time.time()
    backend = "hpc"
    trajectory_path = Path(launch_args.trajectory_sqlite_path)
    hf_model_name = launch_args.hf_model_name
    _emit_status(backend, "starting", "preparing HPC training job folder")
    _tui_state("Training wrapper started")
    _tui_info("Training wrapper started")
    _emit_training_tui_identity(training_config, hf_model_name, trajectory_path)
    artifact_root_dir = resolve_artifact_root_dir(training_config)
    model_cli_name = training_config.model_cli_name
    config_nickname = training_config.config_nickname
    epoch = training_config.epoch
    checkpoints_root = checkpoint_parent_dir(
        artifact_root_dir, model_cli_name, config_nickname, epoch
    )
    next_model_root = final_model_output_parent_dir(
        artifact_root_dir, model_cli_name, config_nickname, epoch
    )
    _tui_info(f"Training checkpoints directory: {checkpoints_root / 'checkpoints'}")
    _tui_info(f"Training next model directory: {next_model_root / 'model'}")
    _tui_state(f"Training checkpoints will be written under {checkpoints_root}")
    _ensure_initial_model_if_missing(backend, training_config, hf_model_name)
    with tempfile.TemporaryDirectory(prefix="training_wrapper_job_") as temp_dir:
        job_dir = Path(temp_dir)
        input_dir = job_dir / "input"
        input_dir.mkdir(parents=True, exist_ok=True)
        dst_trajectory = input_dir / "training_trajectories.sqlite"
        shutil.copy2(trajectory_path, dst_trajectory)
        if launch_args.test_sleep_secs > 0:
            cmd = [
                sys.executable,
                "-c",
                f"import time; time.sleep({launch_args.test_sleep_secs})",
            ]
            stdin_payload = None
        else:
            train_process_launch_args = TrainProcessLaunchArgs(
                job_folder_path=str(job_dir),
                training_trajectory_sqlite_path=str(dst_trajectory),
                orchestrator_socket_path=launch_args.orchestrator_socket_path,
            )
            cmd = [
                "uv",
                "run",
                "torchrun",
                "--nproc_per_node",
                str(launch_args.num_gpus),
                "-m",
                "src_py.train.main",
                *model_to_cli_args(train_process_launch_args),
            ]
            stdin_payload = model_to_json_bytes(training_config)
        if _TRAINING_WRAPPER_LOG_PATH is None:
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE if stdin_payload is not None else None,
            )
            if stdin_payload is not None:
                _write_subprocess_stdin(process, stdin_payload)
            _set_active_process(process)
            _emit_status(backend, "running", f"started torchrun with pid={process.pid}")
            _tui_state(f"Training subprocess started (pid={process.pid})")
            return_code = process.wait()
        else:
            with _TRAINING_WRAPPER_LOG_PATH.open("a", encoding="utf-8") as log_handle:
                process = subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE if stdin_payload is not None else None,
                    stdout=log_handle,
                    stderr=log_handle,
                )
                if stdin_payload is not None:
                    _write_subprocess_stdin(process, stdin_payload)
                _set_active_process(process)
                _emit_status(
                    backend,
                    "running",
                    (
                        f"started torchrun with pid={process.pid}; "
                        f"subprocess output redirected to {_TRAINING_WRAPPER_LOG_PATH}"
                    ),
                )
                _tui_state(f"Training subprocess started (pid={process.pid})")
                return_code = process.wait()
        _set_active_process(None)
        duration_secs = time.time() - started_at
        if _TERMINATION_REQUESTED.is_set():
            _tui_error("Training wrapper cancelled by signal")
            _emit_result_error(
                backend,
                "CANCELLED_BY_SIGNAL",
                "training wrapper received SIGTERM/SIGINT",
                duration_secs,
            )
            return 143
        if return_code == 0:
            _tui_state(
                f"Training completed; checkpoint state available under {checkpoints_root}"
            )
            _emit_result_ok(backend, "training completed", duration_secs)
            return 0
        _tui_error(f"Training failed: torchrun exited with code {return_code}")
        _emit_result_error(
            backend,
            "TRAIN_PROCESS_FAILED",
            f"torchrun exited with code {return_code}",
            duration_secs,
        )
        return return_code


def main() -> int:
    _set_process_name("training_wrapper")
    started_at = time.time()
    _install_signal_handlers()
    launch_args = parse_model_args(_build_parser(), TrainingWrapperLaunchArgs)
    backend_name = "hpc"
    _configure_wrapper_log_path(launch_args.wrapper_log_path)
    configure_tui_forwarder(launch_args.orchestrator_socket_path)
    _tui_state("Training wrapper process initialized")
    _tui_info("Training wrapper process initialized")
    _start_parent_watchdog(backend_name)
    if launch_args.num_gpus <= 0:
        _tui_error("Training wrapper failed: --num-gpus must be positive")
        _emit_result_error(
            backend_name,
            "INVALID_NUM_GPUS",
            "--num-gpus must be positive",
            time.time() - started_at,
        )
        return 2

    try:
        training_config = parse_model_stdin(TrainingRequestArgs)
    except (ValidationError, ValueError) as error:
        _tui_error(f"Training wrapper failed: invalid training config stdin: {error}")
        _emit_result_error(
            backend_name,
            "INVALID_TRAINING_CONFIG_STDIN",
            f"invalid training config stdin: {error}",
            time.time() - started_at,
        )
        return 2

    trajectory_path = Path(launch_args.trajectory_sqlite_path)
    if not trajectory_path.exists() or not trajectory_path.is_file():
        _tui_error(
            f"Training wrapper failed: trajectory sqlite not found at {trajectory_path}"
        )
        _emit_result_error(
            backend_name,
            "TRAJECTORY_NOT_FOUND",
            f"trajectory sqlite does not exist: {trajectory_path}",
            time.time() - started_at,
        )
        return 2

    try:
        return _run_hpc_training(launch_args, training_config)
    except Exception as error:  # noqa: BLE001
        _tui_error(f"Training wrapper runtime error: {error}")
        _emit_result_error(
            backend_name,
            "WRAPPER_RUNTIME_ERROR",
            str(error),
            time.time() - started_at,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
