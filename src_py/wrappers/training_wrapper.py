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

from src_py.load_model_to_path import ensure_model_snapshot
from src_py.train.pathing import model_parent_dir, resolve_artifact_root_dir


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


def _collect_modal_activity_metrics(target: Any, phase: str) -> dict[str, Any]:
    stats_method_names = (
        "get_current_stats",
        "get_stats",
        "stats",
        "current_stats",
    )
    for name in stats_method_names:
        method = getattr(target, name, None)
        if callable(method):
            try:
                value = method()
                if isinstance(value, dict):
                    return {"phase": phase, "available": True, "raw": value}
                return {
                    "phase": phase,
                    "available": True,
                    "raw": str(value),
                }
            except Exception as error:  # noqa: BLE001
                return {
                    "phase": phase,
                    "available": False,
                    "error": f"stats method {name} failed: {error}",
                }
    return {
        "phase": phase,
        "available": False,
        "error": "no supported stats method on modal target",
    }


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


def _configure_wrapper_log_path(raw_path: str) -> None:
    global _TRAINING_WRAPPER_LOG_PATH
    path = raw_path.strip()
    if not path:
        _TRAINING_WRAPPER_LOG_PATH = None
        return
    resolved = Path(path)
    if resolved.parent and not resolved.parent.exists():
        resolved.parent.mkdir(parents=True, exist_ok=True)
    _TRAINING_WRAPPER_LOG_PATH = resolved


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
    parser.add_argument("--hf-model-name", type=str, required=True)
    parser.add_argument("--modal-app-name", type=str, default="credit-assignment-training-service")
    parser.add_argument("--modal-class-name", type=str, default="ExperimentService")
    parser.add_argument("--training-wrapper-log-path", type=str, default="")
    parser.add_argument(
        "--test-sleep-secs",
        type=float,
        default=0.0,
        help="Test-only: run a dummy sleep process instead of real training",
    )
    return parser


def _ensure_initial_model_if_missing(
    backend: str,
    training_config: dict[str, Any],
    hf_model_name: str,
) -> None:
    epoch = int(training_config.get("epoch", 0))
    if epoch != 0:
        return
    artifact_root_dir = resolve_artifact_root_dir(training_config)
    model_cli_name = str(training_config["model_cli_name"])
    config_nickname = str(training_config["config_nickname"])
    parent_dir = model_parent_dir(artifact_root_dir, model_cli_name, config_nickname, epoch)
    model_dir = parent_dir / "model"
    if model_dir.exists():
        return
    _emit_status(
        backend,
        "starting",
        f"initial model missing at {model_dir}; downloading {hf_model_name}",
    )
    ensure_model_snapshot(parent_dir, hf_model_name)


def _run_hpc_training(
    num_gpus: int,
    training_config: dict[str, Any],
    trajectory_path: Path,
    hf_model_name: str,
    test_sleep_secs: float,
) -> int:
    started_at = time.time()
    backend = "hpc"
    _emit_status(backend, "starting", "preparing HPC training job folder")
    _ensure_initial_model_if_missing(backend, training_config, hf_model_name)
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
        if _TRAINING_WRAPPER_LOG_PATH is None:
            process = subprocess.Popen(cmd)
            _set_active_process(process)
            _emit_status(backend, "running", f"started torchrun with pid={process.pid}")
            return_code = process.wait()
        else:
            with _TRAINING_WRAPPER_LOG_PATH.open("a", encoding="utf-8") as log_handle:
                process = subprocess.Popen(cmd, stdout=log_handle, stderr=log_handle)
                _set_active_process(process)
                _emit_status(
                    backend,
                    "running",
                    (
                        f"started torchrun with pid={process.pid}; "
                        f"subprocess output redirected to {_TRAINING_WRAPPER_LOG_PATH}"
                    ),
                )
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
    class_name: str,
    num_gpus: int,
    training_config: dict[str, Any],
    trajectory_path: Path,
    hf_model_name: str,
    test_sleep_secs: float,
) -> int:
    started_at = time.time()
    backend = "modal"
    _ensure_initial_model_if_missing(backend, training_config, hf_model_name)
    model_cli_name = str(training_config["model_cli_name"])
    config_nickname = str(training_config["config_nickname"])
    epoch = int(training_config["epoch"])

    deployed_app_name: str | None = None
    if test_sleep_secs <= 0:
        app_name_result, deploy_code = ensure_deployed(
            model_cli_name=model_cli_name,
            model_api_name=hf_model_name,
            config_nickname=config_nickname,
            epoch=epoch,
            num_gpus=num_gpus,
        )
        if deploy_code != 0:
            _emit_result_error(
                backend,
                "MODAL_DEPLOY_FAILED",
                f"failed to deploy modal training app {app_name_result}",
                time.time() - started_at,
            )
            return deploy_code
        deployed_app_name = app_name_result

    if app_name and deployed_app_name and app_name != deployed_app_name:
        _emit_status(
            backend,
            "starting",
            (
                f"ignoring --modal-app-name={app_name}; using wrapper-managed deployment "
                f"name={deployed_app_name}"
            ),
        )
    target_app_name = deployed_app_name or app_name

    try:
        _emit_status(
            backend,
            "starting",
            (
                f"submitting modal training class call: {class_name} "
                f"for experiment {model_cli_name}_{config_nickname}_e{epoch} with num_gpus={num_gpus}"
            ),
        )
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

        cls = modal.Cls.from_name(target_app_name, class_name)
        instance = cls()
        trajectory_bytes = trajectory_path.read_bytes()
        result_queue: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=1)

        def _invoke() -> None:
            try:
                response = instance.train.remote(training_config, trajectory_bytes, num_gpus)
                result_queue.put(("ok", response))
            except Exception as error:  # noqa: BLE001
                result_queue.put(("error", str(error)))

        worker = threading.Thread(target=_invoke, daemon=True)
        before_metrics = _collect_modal_activity_metrics(instance.train, "before_call")
        _emit_status_with_metrics(
            backend,
            "modal_call_pre",
            "collected modal activity metrics before train call",
            before_metrics,
        )
        worker.start()
        while worker.is_alive():
            if _TERMINATION_REQUESTED.is_set():
                exit_metrics = _collect_modal_activity_metrics(instance.train, "wrapper_exit")
                _emit_status_with_metrics(
                    backend,
                    "modal_wrapper_exit",
                    "collected modal activity metrics before training wrapper exit",
                    exit_metrics,
                )
                _emit_result_error(
                    backend,
                    "CANCELLED_BY_SIGNAL",
                    "training wrapper received SIGTERM/SIGINT",
                    time.time() - started_at,
                )
                return 143
            worker.join(timeout=0.5)

        if result_queue.empty():
            after_metrics = _collect_modal_activity_metrics(instance.train, "after_call")
            _emit_status_with_metrics(
                backend,
                "modal_call_post",
                "collected modal activity metrics after train call",
                after_metrics,
            )
            exit_metrics = _collect_modal_activity_metrics(instance.train, "wrapper_exit")
            _emit_status_with_metrics(
                backend,
                "modal_wrapper_exit",
                "collected modal activity metrics before training wrapper exit",
                exit_metrics,
            )
            _emit_result_error(
                backend,
                "MODAL_TRAIN_NO_RESULT",
                "modal function call finished without result",
                time.time() - started_at,
            )
            return 1

        kind, payload = result_queue.get_nowait()
        if kind == "error":
            after_metrics = _collect_modal_activity_metrics(instance.train, "after_call")
            _emit_status_with_metrics(
                backend,
                "modal_call_post",
                "collected modal activity metrics after train call",
                after_metrics,
            )
            exit_metrics = _collect_modal_activity_metrics(instance.train, "wrapper_exit")
            _emit_status_with_metrics(
                backend,
                "modal_wrapper_exit",
                "collected modal activity metrics before training wrapper exit",
                exit_metrics,
            )
            _emit_result_error(
                backend,
                "MODAL_TRAIN_REMOTE_ERROR",
                str(payload),
                time.time() - started_at,
            )
            return 1
        response = payload
        after_metrics = _collect_modal_activity_metrics(instance.train, "after_call")
        _emit_status_with_metrics(
            backend,
            "modal_call_post",
            "collected modal activity metrics after train call",
            after_metrics,
        )
        exit_metrics = _collect_modal_activity_metrics(instance.train, "wrapper_exit")
        _emit_status_with_metrics(
            backend,
            "modal_wrapper_exit",
            "collected modal activity metrics before training wrapper exit",
            exit_metrics,
        )
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
    finally:
        if deployed_app_name is not None:
            app_name_result, undeploy_code = ensure_undeployed(
                model_cli_name=model_cli_name,
                model_api_name=hf_model_name,
                config_nickname=config_nickname,
                epoch=epoch,
            )
            if undeploy_code != 0:
                _emit_status(
                    backend,
                    "stopping",
                    f"failed to undeploy modal training app {app_name_result}",
                )


def main() -> int:
    _set_process_name("training_wrapper")
    started_at = time.time()
    _install_signal_handlers()
    args = _build_parser().parse_args()
    _configure_wrapper_log_path(args.training_wrapper_log_path)
    _start_parent_watchdog(args.backend)
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
        if args.backend == "modal":
            _emit_status(
                "modal",
                "starting",
                "using local HPC-style training path; wrapper-managed modal app deployment disabled",
            )
        return _run_hpc_training(
            args.num_gpus,
            training_config,
            trajectory_path,
            args.hf_model_name,
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
