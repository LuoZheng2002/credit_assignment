from __future__ import annotations

import argparse
import ctypes
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from src_py.load_model_to_path import ensure_model_snapshot
from src_py.train.cli_args import (
    SftTrainProcessLaunchArgs,
    SftWrapperLaunchArgs,
    TrainingRequestArgs,
    add_model_arguments,
    model_to_cli_args,
    parse_model_args,
    parse_model_stdin,
    write_model_json_file,
)
from src_py.tui_logging import (
    _tui_error,
    _tui_info,
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
_SFT_WRAPPER_LOG_PATH: Path | None = None
_WRAPPER_LOG_FILE_HANDLE: Any | None = None


def _configure_wrapper_log_path(raw_path: str) -> None:
    global _SFT_WRAPPER_LOG_PATH, _WRAPPER_LOG_FILE_HANDLE
    path = raw_path.strip()
    if not path:
        raise ValueError("--wrapper-log-path must be non-empty")
    resolved = Path(path).expanduser().resolve()
    if resolved.parent and not resolved.parent.exists():
        resolved.parent.mkdir(parents=True, exist_ok=True)
    _SFT_WRAPPER_LOG_PATH = resolved
    _WRAPPER_LOG_FILE_HANDLE = open(resolved, "a", buffering=1, encoding="utf-8")
    os.dup2(_WRAPPER_LOG_FILE_HANDLE.fileno(), 1)
    os.dup2(_WRAPPER_LOG_FILE_HANDLE.fileno(), 2)
    sys.stdout = os.fdopen(1, "w", buffering=1, encoding="utf-8", closefd=False)
    sys.stderr = os.fdopen(2, "w", buffering=1, encoding="utf-8", closefd=False)


def _emit_sft_tui_identity(
    training_config: TrainingRequestArgs, hf_model_name: str, sft_data_path: Path
) -> None:
    _tui_info(
        "SFT wrapper config: "
        f"model_cli_name={training_config.model_cli_name} "
        f"config_nickname={training_config.config_nickname} "
        f"hf_model_name={hf_model_name} "
        f"sft_training_data_path={sft_data_path}"
    )


def _set_active_process(process: subprocess.Popen[Any] | None) -> None:
    global _ACTIVE_PROCESS
    with _ACTIVE_PROCESS_LOCK:
        _ACTIVE_PROCESS = process


def _on_signal(signum: int, _: Any) -> None:
    _TERMINATION_REQUESTED.set()
    _tui_error(f"SFT wrapper received signal {signum}")
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
                        f"(initial_ppid={parent_pid}, current_ppid={current_parent}); cancelling SFT wrapper"
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
        description="SFT wrapper that launches SFT training"
    )
    add_model_arguments(parser, SftWrapperLaunchArgs)
    return parser


def _ensure_initial_model_if_missing(
    backend: str,
    training_config: TrainingRequestArgs,
    hf_model_name: str,
) -> None:
    parent_dir = Path(training_config.model_parent_dir)
    model_dir = parent_dir / "model"
    if model_dir.exists():
        return
    _emit_status(
        backend,
        "starting",
        f"initial model missing at {model_dir}; downloading {hf_model_name}",
    )
    ensure_model_snapshot(parent_dir, hf_model_name)


def _sft_request_json_path(training_config: TrainingRequestArgs) -> Path:
    checkpoints_root = Path(training_config.checkpoints_parent_dir)
    checkpoints_root.mkdir(parents=True, exist_ok=True)
    return checkpoints_root / "sft_request.json"


def _run_sft_training(
    launch_args: SftWrapperLaunchArgs,
    training_config: TrainingRequestArgs,
) -> int:
    started_at = time.time()
    backend = "sft"
    sft_data_path = Path(launch_args.sft_training_data_path)
    hf_model_name = launch_args.hf_model_name
    _emit_status(backend, "starting", "preparing SFT training")
    _tui_info("SFT wrapper started")
    _emit_sft_tui_identity(training_config, hf_model_name, sft_data_path)
    checkpoints_root = Path(training_config.checkpoints_parent_dir)
    next_model_root = Path(training_config.final_model_output_parent_dir)
    _tui_info(f"SFT checkpoints directory: {checkpoints_root / 'checkpoints'}")
    _tui_info(f"SFT output model directory: {next_model_root / 'model'}")
    _ensure_initial_model_if_missing(backend, training_config, hf_model_name)

    request_json_path = _sft_request_json_path(training_config)
    write_model_json_file(training_config, request_json_path)
    train_process_launch_args = SftTrainProcessLaunchArgs(
        sft_training_data_path=str(sft_data_path),
        training_request_json_path=str(request_json_path),
        orchestrator_socket_path=launch_args.orchestrator_socket_path,
    )
    cmd = [
        "uv",
        "run",
        "torchrun",
        "--nproc_per_node",
        str(launch_args.num_gpus),
        "-m",
        "src_py.train.sft_main",
        *model_to_cli_args(train_process_launch_args),
    ]
    try:
        if _SFT_WRAPPER_LOG_PATH is None:
            process = subprocess.Popen(
                cmd,
                stdin=None,
            )
            _set_active_process(process)
            _emit_status(backend, "running", f"started torchrun with pid={process.pid}")
            _tui_info(f"SFT training subprocess started (pid={process.pid})")
            return_code = process.wait()
        else:
            with _SFT_WRAPPER_LOG_PATH.open("a", encoding="utf-8") as log_handle:
                process = subprocess.Popen(
                    cmd,
                    stdin=None,
                    stdout=log_handle,
                    stderr=log_handle,
                )
                _set_active_process(process)
                _emit_status(
                    backend,
                    "running",
                    (
                        f"started torchrun with pid={process.pid}; "
                        f"subprocess output redirected to {_SFT_WRAPPER_LOG_PATH}"
                    ),
                )
                _tui_info(f"SFT training subprocess started (pid={process.pid})")
                return_code = process.wait()
    finally:
        _set_active_process(None)
        if request_json_path is not None and request_json_path.exists():
            request_json_path.unlink()
    duration_secs = time.time() - started_at
    if _TERMINATION_REQUESTED.is_set():
        _tui_error("SFT wrapper cancelled by signal")
        _emit_result_error(
            backend,
            "CANCELLED_BY_SIGNAL",
            "SFT wrapper received SIGTERM/SIGINT",
            duration_secs,
        )
        return 143
    if return_code == 0:
        _tui_info(f"SFT completed; model output available under {next_model_root}")
        _emit_result_ok(backend, "SFT training completed", duration_secs)
        return 0
    _tui_error(f"SFT failed: torchrun exited with code {return_code}")
    _emit_result_error(
        backend,
        "TRAIN_PROCESS_FAILED",
        f"torchrun exited with code {return_code}",
        duration_secs,
    )
    return return_code


def main() -> int:
    _set_process_name("sft_wrapper")
    started_at = time.time()
    _install_signal_handlers()
    launch_args = parse_model_args(_build_parser(), SftWrapperLaunchArgs)
    backend_name = "sft"
    _configure_wrapper_log_path(launch_args.wrapper_log_path)
    configure_tui_forwarder(launch_args.orchestrator_socket_path)
    _tui_info("SFT wrapper process initialized")
    _start_parent_watchdog(backend_name)
    if launch_args.num_gpus <= 0:
        _tui_error("SFT wrapper failed: --num-gpus must be positive")
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
        _tui_error(f"SFT wrapper failed: invalid training config stdin: {error}")
        _emit_result_error(
            backend_name,
            "INVALID_TRAINING_CONFIG_STDIN",
            f"invalid training config stdin: {error}",
            time.time() - started_at,
        )
        return 2

    sft_data_path = Path(launch_args.sft_training_data_path)
    if not sft_data_path.exists() or not sft_data_path.is_file():
        _tui_error(
            f"SFT wrapper failed: SFT training data not found at {sft_data_path}"
        )
        _emit_result_error(
            backend_name,
            "SFT_DATA_NOT_FOUND",
            f"SFT training data does not exist: {sft_data_path}",
            time.time() - started_at,
        )
        return 2

    try:
        return _run_sft_training(launch_args, training_config)
    except Exception as error:  # noqa: BLE001
        _tui_error(f"SFT wrapper runtime error: {error}")
        _emit_result_error(
            backend_name,
            "WRAPPER_RUNTIME_ERROR",
            str(error),
            time.time() - started_at,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
