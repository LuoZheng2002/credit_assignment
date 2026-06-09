from __future__ import annotations

import argparse
import json
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any


def _emit_event(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=True), flush=True)


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


def _set_active_process(process: subprocess.Popen[Any] | None) -> None:
    global _ACTIVE_PROCESS
    with _ACTIVE_PROCESS_LOCK:
        _ACTIVE_PROCESS = process


def _on_signal(signum: int, _: Any) -> None:
    _TERMINATION_REQUESTED.set()
    _emit_status("wrapper", "cancelling", f"received signal {signum}")
    with _ACTIVE_PROCESS_LOCK:
        process = _ACTIVE_PROCESS
    if process is not None and process.poll() is None:
        process.terminate()


def _install_signal_handlers() -> None:
    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)


def _to_toml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    raise ValueError(f"Unsupported value type for train_request.toml: {type(value)}")


def _write_flat_toml(path: Path, payload: dict[str, Any]) -> None:
    lines: list[str] = []
    for key in sorted(payload.keys()):
        value = payload[key]
        if value is None:
            continue
        if isinstance(value, (dict, list)):
            raise ValueError(f"Nested key is not supported in train_request.toml: {key}")
        lines.append(f"{key} = {_to_toml_scalar(value)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Training wrapper for HPC and Modal backends")
    parser.add_argument("--backend", choices=["hpc", "modal"], required=True)
    parser.add_argument("--num-gpus", type=int, required=True)
    parser.add_argument("--training-config-json", type=str, required=True)
    parser.add_argument("--trajectory-sqlite-path", type=str, required=True)
    parser.add_argument("--modal-app-name", type=str, default="credit-assignment-sglang-service")
    parser.add_argument("--modal-function-name", type=str, default="modal_train")
    parser.add_argument(
        "--test-sleep-secs",
        type=float,
        default=0.0,
        help="Test-only: run a dummy sleep process instead of real training",
    )
    return parser


def _run_hpc_training(
    num_gpus: int,
    training_config: dict[str, Any],
    trajectory_path: Path,
    test_sleep_secs: float,
) -> int:
    started_at = time.time()
    backend = "hpc"
    _emit_status(backend, "starting", "preparing HPC training job folder")
    with tempfile.TemporaryDirectory(prefix="training_wrapper_job_") as temp_dir:
        job_dir = Path(temp_dir)
        input_dir = job_dir / "input"
        input_dir.mkdir(parents=True, exist_ok=True)
        dst_trajectory = input_dir / "training_trajectories.sqlite"
        shutil.copy2(trajectory_path, dst_trajectory)
        _write_flat_toml(job_dir / "train_request.toml", training_config)

        if test_sleep_secs > 0:
            cmd = [
                sys.executable,
                "-c",
                f"import time; time.sleep({test_sleep_secs})",
            ]
        else:
            cmd = [
                "uv",
                "run",
                "torchrun",
                "--nproc_per_node",
                str(num_gpus),
                "-m",
                "src_py.train.main_from_config",
                "--job-folder-path",
                str(job_dir),
            ]
        process = subprocess.Popen(cmd)
        _set_active_process(process)
        _emit_status(backend, "running", f"started torchrun with pid={process.pid}")
        return_code = process.wait()
        _set_active_process(None)
        duration_secs = time.time() - started_at
        if _TERMINATION_REQUESTED.is_set():
            _emit_result_error(
                backend,
                "CANCELLED_BY_SIGNAL",
                "training wrapper received SIGTERM/SIGINT",
                duration_secs,
            )
            return 143
        if return_code == 0:
            _emit_result_ok(backend, "training completed", duration_secs)
            return 0
        _emit_result_error(
            backend,
            "TRAIN_PROCESS_FAILED",
            f"torchrun exited with code {return_code}",
            duration_secs,
        )
        return return_code


def _run_modal_training(
    app_name: str,
    function_name: str,
    num_gpus: int,
    training_config: dict[str, Any],
    trajectory_path: Path,
    test_sleep_secs: float,
) -> int:
    started_at = time.time()
    backend = "modal"
    if num_gpus != 1:
        _emit_result_error(
            backend,
            "MODAL_GPU_COUNT_UNSUPPORTED",
            f"modal backend requires --num-gpus=1, got {num_gpus}",
            time.time() - started_at,
        )
        return 2

    _emit_status(backend, "starting", "submitting modal training function call")
    import queue

    if test_sleep_secs > 0:
        result_queue: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=1)

        def _invoke_test() -> None:
            try:
                time.sleep(test_sleep_secs)
                result_queue.put(("ok", {"ok": True}))
            except Exception as error:  # noqa: BLE001
                result_queue.put(("error", str(error)))

        worker = threading.Thread(target=_invoke_test, daemon=True)
        worker.start()
        while worker.is_alive():
            if _TERMINATION_REQUESTED.is_set():
                _emit_result_error(
                    backend,
                    "CANCELLED_BY_SIGNAL",
                    "training wrapper received SIGTERM/SIGINT",
                    time.time() - started_at,
                )
                return 143
            worker.join(timeout=0.5)
        _emit_result_ok(backend, "modal training completed", time.time() - started_at)
        return 0

    import modal

    function = modal.Function.from_name(app_name, function_name)
    trajectory_bytes = trajectory_path.read_bytes()
    result_queue: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=1)

    def _invoke() -> None:
        try:
            response = function.remote(training_config, trajectory_bytes, 1)
            result_queue.put(("ok", response))
        except Exception as error:  # noqa: BLE001
            result_queue.put(("error", str(error)))

    worker = threading.Thread(target=_invoke, daemon=True)
    worker.start()
    while worker.is_alive():
        if _TERMINATION_REQUESTED.is_set():
            _emit_result_error(
                backend,
                "CANCELLED_BY_SIGNAL",
                "training wrapper received SIGTERM/SIGINT",
                time.time() - started_at,
            )
            return 143
        worker.join(timeout=0.5)

    if result_queue.empty():
        _emit_result_error(
            backend,
            "MODAL_TRAIN_NO_RESULT",
            "modal function call finished without result",
            time.time() - started_at,
        )
        return 1

    kind, payload = result_queue.get_nowait()
    if kind == "error":
        _emit_result_error(
            backend,
            "MODAL_TRAIN_REMOTE_ERROR",
            str(payload),
            time.time() - started_at,
        )
        return 1
    response = payload
    duration_secs = time.time() - started_at
    if response.get("ok", False):
        _emit_result_ok(backend, "modal training completed", duration_secs)
        return 0
    _emit_result_error(
        backend,
        str(response.get("error_code") or "MODAL_TRAIN_FAILED"),
        str(response.get("error") or response.get("error_message") or "modal training failed"),
        duration_secs,
    )
    return 1


def main() -> int:
    started_at = time.time()
    _install_signal_handlers()
    args = _build_parser().parse_args()
    if args.num_gpus <= 0:
        _emit_result_error(
            args.backend,
            "INVALID_NUM_GPUS",
            "--num-gpus must be positive",
            time.time() - started_at,
        )
        return 2

    try:
        training_config = json.loads(args.training_config_json)
    except json.JSONDecodeError as error:
        _emit_result_error(
            args.backend,
            "INVALID_TRAINING_CONFIG_JSON",
            f"invalid --training-config-json: {error}",
            time.time() - started_at,
        )
        return 2
    trajectory_path = Path(args.trajectory_sqlite_path)
    if not trajectory_path.exists() or not trajectory_path.is_file():
        _emit_result_error(
            args.backend,
            "TRAJECTORY_NOT_FOUND",
            f"trajectory sqlite does not exist: {trajectory_path}",
            time.time() - started_at,
        )
        return 2

    try:
        if args.backend == "hpc":
            return _run_hpc_training(
                args.num_gpus,
                training_config,
                trajectory_path,
                args.test_sleep_secs,
            )

        return _run_modal_training(
            args.modal_app_name,
            args.modal_function_name,
            args.num_gpus,
            training_config,
            trajectory_path,
            args.test_sleep_secs,
        )
    except Exception as error:  # noqa: BLE001
        _emit_result_error(
            args.backend,
            "WRAPPER_RUNTIME_ERROR",
            str(error),
            time.time() - started_at,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
