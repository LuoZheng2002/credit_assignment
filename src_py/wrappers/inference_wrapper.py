from __future__ import annotations

import argparse
import json
import signal
import socketserver
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from src_py.load_model_to_path import ensure_model_snapshot


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


def _post_json(url: str, payload: dict[str, Any], timeout: float = 600.0) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = Request(url=url, method="POST", data=body)
    req.add_header("content-type", "application/json")
    with urlopen(req, timeout=timeout) as response:  # noqa: S310
        return json.loads(response.read().decode("utf-8"))


def _wait_health(url: str, timeout_secs: int) -> None:
    deadline = time.time() + timeout_secs
    while time.time() < deadline:
        try:
            with urlopen(url, timeout=2):  # noqa: S310
                return
        except (HTTPError, URLError):
            time.sleep(1)
    raise TimeoutError(f"timed out waiting for health endpoint: {url}")


class HpcBackend:
    def __init__(self, model_path: str, num_gpus: int, upstream_port: int, epoch: int, hf_model_name: str) -> None:
        model_path_obj = Path(model_path)
        if not model_path_obj.exists() and epoch == 0:
            _emit_status(
                "hpc",
                "starting",
                f"initial model missing at {model_path_obj}; downloading {hf_model_name}",
            )
            ensure_model_snapshot(model_path_obj.parent, hf_model_name)
        if not model_path_obj.exists():
            raise FileNotFoundError(f"model path does not exist: {model_path}")
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
        self._upstream_url = f"http://127.0.0.1:{upstream_port}"
        _emit_status("hpc", "starting", f"launching local sglang on port {upstream_port}")
        _wait_health(f"{self._upstream_url}/health", timeout_secs=300)
        _emit_status("hpc", "ready", "local sglang health check passed")

    def generate(self, payload: dict[str, Any]) -> dict[str, Any]:
        return _post_json(f"{self._upstream_url}/generate", payload)

    def shutdown(self) -> None:
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
        hf_model_name: str,
    ) -> None:
        import modal

        cls = modal.Cls.from_name(app_name, class_name)
        self._instance = cls(
            model_cli_name=model_cli_name,
            config_nickname=config_nickname,
            model_name=hf_model_name,
        )
        _emit_status(
            "modal",
            "ready",
            (
                f"bound modal class {app_name}.{class_name} "
                f"for experiment {model_cli_name}_{config_nickname}"
            ),
        )

    def generate(self, payload: dict[str, Any]) -> dict[str, Any]:
        before_metrics = _collect_modal_activity_metrics(self._instance.generate, "before_call")
        _emit_status_with_metrics(
            "modal",
            "modal_call_pre",
            "collected modal activity metrics before generate call",
            before_metrics,
        )
        result = self._instance.generate.remote(payload)
        after_metrics = _collect_modal_activity_metrics(self._instance.generate, "after_call")
        _emit_status_with_metrics(
            "modal",
            "modal_call_post",
            "collected modal activity metrics after generate call",
            after_metrics,
        )
        return result

    def shutdown(self) -> None:
        metrics = _collect_modal_activity_metrics(self._instance.generate, "wrapper_exit")
        _emit_status_with_metrics(
            "modal",
            "modal_wrapper_exit",
            "collected modal activity metrics before inference wrapper exit",
            metrics,
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
    parser.add_argument("--modal-app-name", type=str, default="credit-assignment-inference-service")
    parser.add_argument("--modal-class-name", type=str, default="ExperimentService")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    if args.listen_port <= 0:
        raise ValueError("--listen-port must be positive")

    if args.backend == "hpc":
        if not args.model_path:
            raise ValueError("--model-path is required for --backend=hpc")
        backend = HpcBackend(
            args.model_path,
            args.num_gpus,
            args.listen_port + 1,
            args.epoch,
            args.hf_model_name,
        )
    else:
        backend = ModalBackend(
            args.modal_app_name,
            args.modal_class_name,
            args.model_cli_name,
            args.config_nickname,
            args.hf_model_name,
        )

    _emit_status(args.backend, "starting", f"binding local wrapper server on port {args.listen_port}")

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path != "/health":
                self.send_response(404)
                self.end_headers()
                return
            body = json.dumps({"status": "ok"}).encode("utf-8")
            self.send_response(200)
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
    _emit_status(args.backend, "ready", "inference wrapper server is ready")

    def _shutdown(*_: Any) -> None:
        _emit_status(args.backend, "stopping", "received termination signal")
        backend.shutdown()
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)
    try:
        server.serve_forever()
    finally:
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


if __name__ == "__main__":
    raise SystemExit(main())
