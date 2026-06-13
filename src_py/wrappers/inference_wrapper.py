from __future__ import annotations

import argparse
import ctypes
import json
import os
import signal
import socketserver
import subprocess
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def _early_redirect_to_log(argv: list[str]) -> None:
    """Redirect stdout/stderr to the wrapper log file before any application imports.

    Scanning argv directly (without argparse) keeps this function stdlib-only so that
    any import error in an application dependency is captured in the log rather than
    silently swallowed by the Stdio::null() pipes set by the Rust launcher.
    """
    log_path: str | None = None
    for i, arg in enumerate(argv):
        if arg == "--wrapper-log-path" and i + 1 < len(argv):
            log_path = argv[i + 1]
            break
        if arg.startswith("--wrapper-log-path="):
            log_path = arg[len("--wrapper-log-path=") :]
            break
    if not log_path or not log_path.strip():
        return
    path = Path(log_path.strip()).expanduser().resolve()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = open(path, "a", buffering=1, encoding="utf-8")  # noqa: SIM115
        os.dup2(handle.fileno(), 1)
        os.dup2(handle.fileno(), 2)
        sys.stdout = os.fdopen(1, "w", buffering=1, encoding="utf-8", closefd=False)
        sys.stderr = os.fdopen(2, "w", buffering=1, encoding="utf-8", closefd=False)
    except Exception:  # noqa: BLE001
        pass  # best-effort; _configure_wrapper_log_file in main() will report the failure


_early_redirect_to_log(sys.argv)

# Application-level imports placed after the early redirect so that any ImportError
# or module-load-time exception is captured in the log file rather than discarded.
from src_py.load_model_to_path import ensure_model_snapshot
from src_py.tui_logging import (
    _tui_error,
    _tui_info,
    configure_tui_forwarder,
)

_WRAPPER_LOG_FILE_HANDLE: Any | None = None


def _set_process_name(name: str) -> None:
    try:
        libc = ctypes.CDLL(None)
        pr_set_name = 15
        libc.prctl(pr_set_name, name.encode("utf-8")[:15], 0, 0, 0)
    except Exception:
        pass


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


def _configure_wrapper_log_file(log_path: str) -> None:
    global _WRAPPER_LOG_FILE_HANDLE
    resolved = log_path.strip()
    if not resolved:
        raise ValueError("--wrapper-log-path must be non-empty")
    path = Path(resolved).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    _WRAPPER_LOG_FILE_HANDLE = open(path, "a", buffering=1, encoding="utf-8")
    os.dup2(_WRAPPER_LOG_FILE_HANDLE.fileno(), 1)
    os.dup2(_WRAPPER_LOG_FILE_HANDLE.fileno(), 2)
    sys.stdout = os.fdopen(1, "w", buffering=1, encoding="utf-8", closefd=False)
    sys.stderr = os.fdopen(2, "w", buffering=1, encoding="utf-8", closefd=False)


def _emit_inference_tui_identity(
    model_cli_name: str,
    config_nickname: str,
    epoch: int,
    hf_model_name: str,
    listen_port: int,
) -> None:
    _tui_info(
        "Inference wrapper config: "
        f"model_cli_name={model_cli_name} "
        f"config_nickname={config_nickname} "
        f"epoch={epoch} "
        f"hf_model_name={hf_model_name} "
        f"listen_port={listen_port}"
    )


def _emit_inference_identity(
    backend: str,
    model_official_name: str,
    config_nickname: str,
    epoch: int,
) -> None:
    _emit_event(
        {
            "type": "status",
            "backend": backend,
            "status": "identity",
            "message": "inference service identity",
            "model_official_name": model_official_name,
            "config_nickname": config_nickname,
            "epoch": epoch,
            "timestamp": time.time(),
        }
    )


def _post_json(
    url: str,
    payload: dict[str, Any],
    timeout: float = 600.0,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = Request(url=url, method="POST", data=body)
    req.add_header("content-type", "application/json")
    if headers:
        for key, value in headers.items():
            req.add_header(key, value)
    with urlopen(req, timeout=timeout) as response:  # noqa: S310
        return json.loads(response.read().decode("utf-8"))


def _get_json(
    url: str,
    timeout: float = 10.0,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    req = Request(url=url, method="GET")
    if headers:
        for key, value in headers.items():
            req.add_header(key, value)
    with urlopen(req, timeout=timeout) as response:  # noqa: S310
        raw = response.read().decode("utf-8").strip()
        if raw == "":
            return {"status": "ok"}
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
            return {"status": "ok", "value": parsed}
        except json.JSONDecodeError:
            return {"status": "ok", "raw": raw[:200]}


def _wait_health(
    url: str, timeout_secs: int, headers: dict[str, str] | None = None
) -> None:
    deadline = time.time() + timeout_secs
    while time.time() < deadline:
        try:
            _get_json(url, timeout=2, headers=headers)
            return
        except (HTTPError, URLError):
            time.sleep(5)
    raise TimeoutError(f"timed out waiting for health endpoint: {url}")


def _probe_health(
    url: str,
    timeout_secs: float = 2.0,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    return _get_json(url, timeout=timeout_secs, headers=headers)


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


def _resolve_sglang_python_executable() -> str:
    override = os.environ.get("SGLANG_PYTHON_BIN", "").strip()
    if override:
        override_path = Path(override)
        if override_path.is_file():
            return str(override_path)
        raise FileNotFoundError(f"SGLANG_PYTHON_BIN does not exist: {override}")

    repo_root = Path(__file__).resolve().parents[2]
    python_bin = repo_root / "pyprojects" / "sglang" / ".venv" / "bin" / "python"
    if python_bin.is_file():
        return str(python_bin)
    raise FileNotFoundError(
        "expected sglang python executable at "
        f"{python_bin}; run 'uv sync --project pyprojects/sglang' first"
    )


def _emit_sglang_kernels_version(sglang_python: str) -> None:
    probe_cmd = [
        sglang_python,
        "-c",
        "import kernels; print(f'{kernels.__version__}|{kernels.__file__}')",
    ]
    result = subprocess.run(
        probe_cmd,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip() or "unknown error"
        _emit_status(
            "hpc",
            "starting",
            f"failed to probe kernels version before sglang launch: {message}",
        )
        return

    details = result.stdout.strip() or "missing version output"
    _emit_status(
        "hpc",
        "starting",
        f"sglang env kernels before launch: {details}",
    )


def _start_parent_watchdog(
    backend_name: str,
    backend: Any,
    server: ThreadedHTTPServer,
    shutdown_event: threading.Event,
    poll_secs: float = 2.0,
) -> threading.Thread:
    parent_pid = os.getppid()

    def _watch() -> None:
        while not shutdown_event.is_set():
            current_parent = os.getppid()
            if current_parent != parent_pid or not _process_exists(parent_pid):
                _emit_status(
                    backend_name,
                    "stopping",
                    f"parent process exited (initial_ppid={parent_pid}, current_ppid={current_parent}); shutting down wrapper",
                )
                try:
                    backend.shutdown()
                except Exception as error:  # noqa: BLE001
                    _emit_event(
                        {
                            "type": "error",
                            "backend": backend_name,
                            "error_code": "PARENT_WATCHDOG_SHUTDOWN_FAILED",
                            "error_message": str(error),
                            "timestamp": time.time(),
                        }
                    )
                threading.Thread(target=server.shutdown, daemon=True).start()
                return
            shutdown_event.wait(timeout=poll_secs)

    watchdog = threading.Thread(target=_watch, daemon=True)
    watchdog.start()
    return watchdog


def _hpc_server_state_path(upstream_port: int) -> Path:
    return Path(f"/tmp/credit_assignment_hpc_server_{upstream_port}.json")


def _read_hpc_server_state(upstream_port: int) -> dict[str, Any] | None:
    path = _hpc_server_state_path(upstream_port)
    if not path.is_file():
        return None
    try:
        raw = path.read_text(encoding="utf-8")
        payload = json.loads(raw)
        if isinstance(payload, dict):
            return payload
    except Exception:
        return None
    return None


def _write_hpc_server_state(
    upstream_port: int,
    model_cli_name: str,
    config_nickname: str,
    epoch: int,
    model_path: str,
    pid: int,
) -> None:
    payload = {
        "model_cli_name": model_cli_name,
        "config_nickname": config_nickname,
        "epoch": epoch,
        "model_path": model_path,
        "pid": pid,
    }
    _hpc_server_state_path(upstream_port).write_text(
        json.dumps(payload, ensure_ascii=True),
        encoding="utf-8",
    )


class HpcBackend:
    def __init__(
        self,
        model_path: str,
        num_gpus: int,
        upstream_port: int,
        epoch: int,
        hf_model_name: str,
        model_cli_name: str,
        config_nickname: str,
        wrapper_log_path: str,
    ) -> None:
        model_path_obj = Path(model_path)
        self._process: subprocess.Popen[bytes] | None = None
        self._upstream_url = f"http://127.0.0.1:{upstream_port}"
        self._wrapper_log_path = Path(wrapper_log_path).expanduser().resolve()
        self._wrapper_log_path.parent.mkdir(parents=True, exist_ok=True)
        if not model_path_obj.exists() and epoch == 0:
            _emit_status(
                "hpc",
                "starting",
                f"initial model missing at {model_path_obj}; downloading {hf_model_name}",
            )
            _tui_info(f"Downloading initial model for inference: {hf_model_name}")
            ensure_model_snapshot(model_path_obj.parent, hf_model_name)
        if not model_path_obj.exists():
            raise FileNotFoundError(f"model path does not exist: {model_path}")
        try:
            _probe_health(f"{self._upstream_url}/health")
            existing_state = _read_hpc_server_state(upstream_port)
            if existing_state is None:
                raise RuntimeError(
                    "existing local sglang server detected but config state file is missing; "
                    "cannot validate config match"
                )
            expected = {
                "model_cli_name": model_cli_name,
                "config_nickname": config_nickname,
                "epoch": epoch,
            }
            observed = {
                "model_cli_name": str(existing_state.get("model_cli_name", "")),
                "config_nickname": str(existing_state.get("config_nickname", "")),
                "epoch": int(existing_state.get("epoch", -1)),
            }
            if observed != expected:
                raise RuntimeError(
                    "existing local sglang server config mismatch; "
                    f"expected={expected}, observed={observed}"
                )
            _emit_status(
                "hpc",
                "ready",
                f"reusing existing local sglang on port {upstream_port} for matching config",
            )
            _tui_info(
                f"Reusing local sglang on port {upstream_port} ({self._upstream_url})"
            )
            return
        except Exception as error:  # noqa: BLE001
            if "mismatch" in str(error) or "cannot validate" in str(error):
                raise
        sglang_python = _resolve_sglang_python_executable()
        _emit_sglang_kernels_version(sglang_python)
        child_env = os.environ.copy()
        child_env.setdefault("PYTHONUNBUFFERED", "1")
        _tui_info(
            f"Launching local sglang on port {upstream_port} ({self._upstream_url})"
        )
        with open(
            self._wrapper_log_path, "a", buffering=1, encoding="utf-8"
        ) as log_handle:
            self._process = subprocess.Popen(
                [
                    sglang_python,
                    "-m",
                    "sglang.launch_server",
                    "--model-path",
                    model_path,
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(upstream_port),
                    "--dp",
                    str(num_gpus),
                ],
                stdout=log_handle,
                stderr=log_handle,
                env=child_env,
            )
        _emit_status(
            "hpc", "starting", f"launching local sglang on port {upstream_port}"
        )
        _wait_health(f"{self._upstream_url}/health", timeout_secs=900)
        _tui_info(f"SGLang server is up at {self._upstream_url}")
        _write_hpc_server_state(
            upstream_port=upstream_port,
            model_cli_name=model_cli_name,
            config_nickname=config_nickname,
            epoch=epoch,
            model_path=model_path,
            pid=self._process.pid,
        )
        _emit_status("hpc", "ready", "local sglang health check passed")

    def health(self) -> dict[str, Any]:
        exit_error = self.unexpected_exit_error()
        if exit_error is not None:
            raise RuntimeError(exit_error)
        return _probe_health(f"{self._upstream_url}/health")

    def generate(self, payload: dict[str, Any]) -> dict[str, Any]:
        exit_error = self.unexpected_exit_error()
        if exit_error is not None:
            raise RuntimeError(exit_error)
        return _post_json(f"{self._upstream_url}/generate", payload)

    def unexpected_exit_error(self) -> str | None:
        if self._process is None:
            return None
        return_code = self._process.poll()
        if return_code is None:
            return None
        return (
            f"local sglang process exited unexpectedly with return code {return_code}"
        )

    def shutdown(self) -> None:
        if self._process is None:
            return
        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self._process.kill()


class ThreadedHTTPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inference wrapper that always launches local sglang"
    )
    parser.add_argument("--listen-port", type=int, required=True)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--model-cli-name", type=str, required=True)
    parser.add_argument("--config-nickname", type=str, required=True)
    parser.add_argument("--hf-model-name", type=str, required=True)
    parser.add_argument("--wrapper-log-path", type=str, required=True)
    parser.add_argument("--orchestrator-socket-path", type=str, default="")
    return parser


def main() -> int:
    _set_process_name("inference_wrapper")
    args = _build_parser().parse_args()
    backend_name = "hpc"
    try:
        _configure_wrapper_log_file(args.wrapper_log_path)
        configure_tui_forwarder(args.orchestrator_socket_path)
        _tui_info("Inference wrapper started")
        _emit_inference_tui_identity(
            args.model_cli_name,
            args.config_nickname,
            args.epoch,
            args.hf_model_name,
            args.listen_port,
        )
        if args.listen_port <= 0:
            raise ValueError("--listen-port must be positive")

        backend = HpcBackend(
            args.model_path,
            args.num_gpus,
            args.listen_port + 1,
            args.epoch,
            args.hf_model_name,
            args.model_cli_name,
            args.config_nickname,
            args.wrapper_log_path,
        )

        _emit_inference_identity(
            backend_name,
            args.hf_model_name,
            args.config_nickname,
            args.epoch,
        )

        _emit_status(
            backend_name,
            "starting",
            f"binding local wrapper server on port {args.listen_port}",
        )
        _tui_info(f"Binding inference wrapper server on port {args.listen_port}")

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                if self.path != "/health":
                    self.send_response(404)
                    self.end_headers()
                    return
                try:
                    result = backend.health()
                    body = json.dumps(result).encode("utf-8")
                    self.send_response(200)
                    self.send_header("content-type", "application/json")
                    self.send_header("content-length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                except Exception as error:  # noqa: BLE001
                    _emit_event(
                        {
                            "type": "error",
                            "backend": backend_name,
                            "error_code": "INFERENCE_HEALTH_FAILED",
                            "error_message": str(error),
                            "timestamp": time.time(),
                        }
                    )
                    body = json.dumps(
                        {"status": "error", "message": str(error)}
                    ).encode("utf-8")
                    self.send_response(503)
                    self.send_header("content-type", "application/json")
                    self.send_header("content-length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)

            def do_POST(self) -> None:  # noqa: N802
                if self.path != "/generate":
                    self.send_response(404)
                    self.end_headers()
                    return
                content_len = int(self.headers.get("content-length", "0"))
                raw = self.rfile.read(content_len)
                try:
                    payload = json.loads(raw.decode("utf-8"))
                    result = backend.generate(payload)
                    body = json.dumps(result).encode("utf-8")
                    self.send_response(200)
                    self.send_header("content-type", "application/json")
                    self.send_header("content-length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                except Exception as error:  # noqa: BLE001
                    _emit_event(
                        {
                            "type": "error",
                            "backend": backend_name,
                            "error_code": "INFERENCE_GENERATE_FAILED",
                            "error_message": str(error),
                            "timestamp": time.time(),
                        }
                    )
                    body = json.dumps({"error": {"message": str(error)}}).encode(
                        "utf-8"
                    )
                    self.send_response(500)
                    self.send_header("content-type", "application/json")
                    self.send_header("content-length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)

            def log_message(self, format: str, *args: Any) -> None:
                sys.stdout.write(f"[INFERENCE_WRAPPER] {format % args}\n")
                sys.stdout.flush()

        server = ThreadedHTTPServer(("127.0.0.1", args.listen_port), Handler)
        shutdown_event = threading.Event()
        fatal_error: list[str] = []
        _start_parent_watchdog(backend_name, backend, server, shutdown_event)

        def _watch_backend_process() -> None:
            while not shutdown_event.is_set():
                exit_error = backend.unexpected_exit_error()
                if exit_error is not None:
                    fatal_error.append(exit_error)
                    _emit_event(
                        {
                            "type": "error",
                            "backend": backend_name,
                            "error_code": "SGLANG_PROCESS_EXITED",
                            "error_message": exit_error,
                            "timestamp": time.time(),
                        }
                    )
                    _tui_error(f"Inference backend exited unexpectedly: {exit_error}")
                    shutdown_event.set()
                    threading.Thread(target=server.shutdown, daemon=True).start()
                    return
                shutdown_event.wait(timeout=1.0)

        threading.Thread(target=_watch_backend_process, daemon=True).start()
        _emit_status(backend_name, "ready", "inference wrapper server is ready")
        _tui_info("Inference wrapper server is ready")

        def _shutdown(*_: Any) -> None:
            shutdown_event.set()
            _emit_status(backend_name, "stopping", "received termination signal")
            backend.shutdown()
            threading.Thread(target=server.shutdown, daemon=True).start()

        signal.signal(signal.SIGTERM, _shutdown)
        signal.signal(signal.SIGINT, _shutdown)
        try:
            server.serve_forever()
        finally:
            shutdown_event.set()
            backend.shutdown()
            server.server_close()
            if fatal_error:
                error_message = fatal_error[-1]
                _tui_error(
                    f"Inference wrapper stopping after backend failure: {error_message}"
                )
                _emit_event(
                    {
                        "type": "result",
                        "backend": backend_name,
                        "ok": False,
                        "error_code": "SGLANG_PROCESS_EXITED",
                        "error_message": error_message,
                        "message": error_message,
                        "timestamp": time.time(),
                    }
                )
                return 1
            _tui_info("Inference wrapper stopped")
            _emit_event(
                {
                    "type": "result",
                    "backend": backend_name,
                    "ok": True,
                    "message": "inference wrapper stopped",
                    "timestamp": time.time(),
                }
            )
        return 0
    except Exception as error:  # noqa: BLE001
        _tui_error(f"Inference wrapper failed to start: {error}")
        _emit_event(
            {
                "type": "error",
                "backend": backend_name,
                "error_code": "INFERENCE_WRAPPER_INIT_FAILED",
                "error_message": str(error),
                "timestamp": time.time(),
            }
        )
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
