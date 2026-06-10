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

from src_py.load_model_to_path import ensure_model_snapshot
from src_py.modal.inference_deployment_common import (
    deployment_name,
    ensure_deployed,
    ensure_undeployed,
)


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


def _emit_status_with_metrics(
    backend: str,
    status: str,
    message: str,
    metrics: dict[str, Any],
) -> None:
    payload = {
        "type": "status",
        "backend": backend,
        "status": status,
        "message": message,
        "metrics": metrics,
        "timestamp": time.time(),
    }
    _emit_event(payload)


def _collect_modal_activity_metrics(target: Any, phase: str) -> dict[str, Any]:
    if target is None:
        return {
            "phase": phase,
            "available": False,
            "error": "modal target unavailable (running with explicit --modal-base-url)",
        }
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


def _wait_health(url: str, timeout_secs: int, headers: dict[str, str] | None = None) -> None:
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
    ) -> None:
        model_path_obj = Path(model_path)
        self._process: subprocess.Popen[bytes] | None = None
        self._upstream_url = f"http://127.0.0.1:{upstream_port}"
        if not model_path_obj.exists() and epoch == 0:
            _emit_status(
                "hpc",
                "starting",
                f"initial model missing at {model_path_obj}; downloading {hf_model_name}",
            )
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
            return
        except Exception as error:  # noqa: BLE001
            if "mismatch" in str(error) or "cannot validate" in str(error):
                raise
        self._process = subprocess.Popen(
            [
                "uv",
                "run",
                "--project",
                "pyprojects/sglang",
                "python",
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
            ]
        )
        _emit_status("hpc", "starting", f"launching local sglang on port {upstream_port}")
        _wait_health(f"{self._upstream_url}/health", timeout_secs=300)
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
        return _probe_health(f"{self._upstream_url}/health")

    def generate(self, payload: dict[str, Any]) -> dict[str, Any]:
        return _post_json(f"{self._upstream_url}/generate", payload)

    def shutdown(self) -> None:
        if self._process is None:
            return
        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self._process.kill()


class ModalBackend:
    def __init__(
        self,
        app_name: str,
        class_name: str,
        model_cli_name: str,
        config_nickname: str,
        epoch: int,
        num_gpus: int,
        hf_model_name: str,
        artifact_root_dir: str,
        modal_base_url: str,
        modal_auth_token_env_var: str,
    ) -> None:
        self._app_name = app_name
        self._class_name = class_name
        self._model_cli_name = model_cli_name
        self._config_nickname = config_nickname
        self._epoch = epoch
        self._num_gpus = num_gpus
        self._hf_model_name = hf_model_name
        self._artifact_root_dir = artifact_root_dir
        self._modal_base_url_override = modal_base_url.strip()
        self._modal_auth_token_env_var = modal_auth_token_env_var.strip()
        self._managed_deployment_name: str | None = None
        if self._modal_base_url_override:
            self._app_name = app_name
        else:
            self._app_name = deployment_name(
                model_cli_name=self._model_cli_name,
                model_api_name=self._hf_model_name,
                config_nickname=self._config_nickname,
                epoch=self._epoch,
            )
            _emit_status("modal", "starting", f"ensuring modal deployment {self._app_name}")
            app_name_result, code = ensure_deployed(
                model_cli_name=self._model_cli_name,
                model_api_name=self._hf_model_name,
                config_nickname=self._config_nickname,
                epoch=self._epoch,
                num_gpus=self._num_gpus,
            )
            if code != 0:
                raise RuntimeError(f"failed to deploy modal inference app {app_name_result}")
            self._managed_deployment_name = app_name_result
        self._cls = None
        self._instance = None
        if not self._modal_base_url_override:
            self._instance = self._build_instance()
        self._modal_base_url = self._resolve_modal_base_url()
        self._session_id = f"{model_cli_name}_{config_nickname}_{epoch}"
        self._headers = {
            "Modal-Session-ID": self._session_id,
        }
        if self._modal_auth_token_env_var:
            token = os.environ.get(self._modal_auth_token_env_var, "").strip()
            if token:
                self._headers["authorization"] = f"Bearer {token}"
                self._headers["modal-key"] = token
        _emit_status(
            "modal",
            "container_policy",
            (
                "modal deployment container policy fixed at "
                f"min_containers=max_containers={self._num_gpus}, gpu_per_container=H100:1"
            ),
        )
        self._wait_remote_ready()
        _emit_status(
            "modal",
            "ready",
            (
                f"bound modal HTTP endpoint {self._modal_base_url} "
                f"for experiment {model_cli_name}_{config_nickname}"
            ),
        )

    def _build_instance(self) -> Any:
        import modal

        self._cls = modal.Cls.from_name(
            self._app_name,
            self._class_name,
        )
        return self._cls()

    def _rebind_instance(self) -> None:
        if self._modal_base_url_override:
            return
        self._instance = self._build_instance()
        self._modal_base_url = self._resolve_modal_base_url()

    def _is_transient_modal_error(self, error: Exception) -> bool:
        message = str(error).lower()
        transient_fragments = (
            "is stopped",
            "cancelled",
            "connection",
            "unavailable",
            "deadline exceeded",
            "temporarily unavailable",
        )
        return any(fragment in message for fragment in transient_fragments)

    def _resolve_modal_base_url(self) -> str:
        if self._modal_base_url_override:
            return self._modal_base_url_override.rstrip("/")
        getter = getattr(self._cls, "_experimental_get_flash_urls", None)
        if getter is not None:
            try:
                urls = getter()
                if isinstance(urls, (list, tuple)) and len(urls) > 0:
                    url = str(urls[0]).strip()
                    if url:
                        return url.rstrip("/")
            except Exception:  # noqa: BLE001
                pass
        raise RuntimeError(
            "failed to resolve Modal HTTP URL via _experimental_get_flash_urls; "
            "deploy app with @modal.experimental.http_server and retry"
        )

    def _wait_remote_ready(self, timeout_secs: int = 900) -> None:
        # Temporary toggle: rely on warmup-only readiness to avoid repeated /health polling
        # during container cold start. Set to "0" to restore health polling immediately.
        if os.environ.get("INFERENCE_WRAPPER_SKIP_MODAL_HEALTH_POLL", "1").strip() not in (
            "0",
            "false",
            "False",
        ):
            self._warmup_remote(timeout_secs=timeout_secs)
            return
        _wait_health(f"{self._modal_base_url}/health", timeout_secs=timeout_secs, headers=self._headers)
        self._warmup_remote(timeout_secs=timeout_secs)

    def _warmup_remote(self, timeout_secs: int = 900) -> None:
        warmup_payload = {
            "text": "ready check",
            "sampling_params": {
                "max_new_tokens": 1,
                "temperature": 0.0,
            },
        }
        deadline = time.time() + timeout_secs
        attempt = 0
        last_error: Exception | None = None
        while time.time() < deadline:
            attempt += 1
            try:
                _post_json(
                    f"{self._modal_base_url}/generate",
                    warmup_payload,
                    timeout=60,
                    headers=self._headers,
                )
                return
            except Exception as error:  # noqa: BLE001
                last_error = error
                is_503 = isinstance(error, HTTPError) and error.code == 503
                if not self._is_transient_modal_error(error) and not is_503:
                    raise
                _emit_status(
                    "modal",
                    "retrying",
                    f"modal warmup /generate attempt {attempt} failed transiently: {error}",
                )
                self._rebind_instance()
                time.sleep(5)
        raise TimeoutError(f"modal warmup /generate timed out after {timeout_secs}s: {last_error}")

    def health(self) -> dict[str, Any]:
        result = _probe_health(f"{self._modal_base_url}/health", headers=self._headers)
        if not isinstance(result, dict) or result.get("status") != "ok":
            raise RuntimeError(f"unexpected modal health response: {result}")
        return result

    def generate(self, payload: dict[str, Any]) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(1, 4):
            try:
                result = _post_json(
                    f"{self._modal_base_url}/generate",
                    payload,
                    headers=self._headers,
                )
                return result
            except Exception as error:  # noqa: BLE001
                last_error = error
                if attempt >= 3 or not self._is_transient_modal_error(error):
                    raise
                _emit_status(
                    "modal",
                    "retrying",
                    f"modal generate attempt {attempt}/3 failed transiently: {error}",
                )
                self._rebind_instance()
                self._wait_remote_ready(timeout_secs=300)
        raise RuntimeError(f"modal generate failed after retries: {last_error}")

    def shutdown(self) -> None:
        _emit_status("modal", "stopping", "wrapper shutdown requested")
        if self._managed_deployment_name is not None:
            app_name, code = ensure_undeployed(
                model_cli_name=self._model_cli_name,
                model_api_name=self._hf_model_name,
                config_nickname=self._config_nickname,
                epoch=self._epoch,
            )
            if code != 0:
                _emit_status(
                    "modal",
                    "warning",
                    f"failed to undeploy modal inference app {app_name}",
                )
        return


class ThreadedHTTPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inference wrapper for HPC and Modal backends")
    parser.add_argument("--backend", choices=["hpc", "modal"], required=True)
    parser.add_argument("--listen-port", type=int, required=True)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--model-path", type=str)
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--model-cli-name", type=str, required=True)
    parser.add_argument("--config-nickname", type=str, required=True)
    parser.add_argument("--hf-model-name", type=str, required=True)
    parser.add_argument("--artifact-root-dir", type=str, default="/mnt/service-state")
    parser.add_argument("--modal-app-name", type=str, default="credit-assignment-inference-service")
    parser.add_argument("--modal-class-name", type=str, default="ExperimentService")
    parser.add_argument("--modal-base-url", type=str, default="")
    parser.add_argument("--modal-auth-token-env-var", type=str, default="")
    return parser


def main() -> int:
    _set_process_name("inference_wrapper")
    args = _build_parser().parse_args()
    try:
        if args.listen_port <= 0:
            raise ValueError("--listen-port must be positive")

        if not args.model_path:
            raise ValueError("--model-path is required for both --backend=hpc and --backend=modal")
        if args.backend == "modal":
            _emit_status(
                "modal",
                "starting",
                "using local HPC-style inference path; wrapper-managed modal app deployment disabled",
            )
        backend = HpcBackend(
            args.model_path,
            args.num_gpus,
            args.listen_port + 1,
            args.epoch,
            args.hf_model_name,
            args.model_cli_name,
            args.config_nickname,
        )

        _emit_inference_identity(
            args.backend,
            args.hf_model_name,
            args.config_nickname,
            args.epoch,
        )

        _emit_status(args.backend, "starting", f"binding local wrapper server on port {args.listen_port}")

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
                            "backend": args.backend,
                            "error_code": "INFERENCE_HEALTH_FAILED",
                            "error_message": str(error),
                            "timestamp": time.time(),
                        }
                    )
                    body = json.dumps({"status": "error", "message": str(error)}).encode("utf-8")
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
                            "backend": args.backend,
                            "error_code": "INFERENCE_GENERATE_FAILED",
                            "error_message": str(error),
                            "timestamp": time.time(),
                        }
                    )
                    body = json.dumps({"error": {"message": str(error)}}).encode("utf-8")
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
        _start_parent_watchdog(args.backend, backend, server, shutdown_event)
        _emit_status(args.backend, "ready", "inference wrapper server is ready")

        def _shutdown(*_: Any) -> None:
            shutdown_event.set()
            _emit_status(args.backend, "stopping", "received termination signal")
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
            _emit_event(
                {
                    "type": "result",
                    "backend": args.backend,
                    "ok": True,
                    "message": "inference wrapper stopped",
                    "timestamp": time.time(),
                }
            )
        return 0
    except Exception as error:  # noqa: BLE001
        _emit_event(
            {
                "type": "error",
                "backend": args.backend,
                "error_code": "INFERENCE_WRAPPER_INIT_FAILED",
                "error_message": str(error),
                "timestamp": time.time(),
            }
        )
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
