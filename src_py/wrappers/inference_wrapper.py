from __future__ import annotations

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

from pydantic import BaseModel, ConfigDict

# Application-level imports placed after the early redirect so that any ImportError
# or module-load-time exception is captured in the log file rather than discarded.
from src_py.load_model_to_path import ensure_model_snapshot
from research_utility.connect_rust_parent import (
    RustParentConnection,
    read_orchestrator_socket_path,
)

_WRAPPER_LOG_FILE_HANDLE: Any | None = None

# Module-level reference to the Rust parent connection so that signal handlers,
# the HpcBackend class, and nested functions can send TUI messages.
_CONN: RustParentConnection[Any] | None = None


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


def _emit_inference_text_identity(
    model_cli_name: str,
    config_nickname: str,
    epoch: int,
    hf_model_name: str,
    listen_port: int,
) -> None:
    if _CONN is not None:
        _CONN.send_info(
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


def _post_json_accept_any_response(
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
        raw = response.read().decode("utf-8").strip()
        if raw == "":
            return {"status": "ok"}
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
            return {"status": "ok", "value": parsed}
        except json.JSONDecodeError:
            return {"status": "ok", "message": raw}


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


def _tail_text(path: Path, max_lines: int = 120) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as error:
        return f"<failed to read {path}: {error}>"
    return "\n".join(lines[-max_lines:])


def _wait_health_or_process_exit(
    url: str,
    process: subprocess.Popen[bytes],
    timeout_secs: int,
    log_path: Path,
    headers: dict[str, str] | None = None,
) -> None:
    deadline = time.time() + timeout_secs
    last_error = ""
    while time.time() < deadline:
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(
                "vLLM process exited before health check passed "
                f"(pid={process.pid}, return_code={return_code}). "
                f"Last log lines:\n{_tail_text(log_path)}"
            )
        try:
            _get_json(url, timeout=2, headers=headers)
            return
        except (HTTPError, URLError) as error:
            last_error = repr(error)
            time.sleep(5)
    raise TimeoutError(
        f"timed out waiting {timeout_secs}s for health endpoint: {url}; "
        f"pid={process.pid}; last_probe_error={last_error}; "
        f"process_return_code={process.poll()}; last log lines:\n{_tail_text(log_path)}"
    )


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


def _resolve_vllm_python_executable() -> str:
    override = os.environ.get("VLLM_PYTHON_BIN", "").strip()
    if override:
        override_path = Path(override)
        if override_path.is_file():
            return str(override_path)
        raise FileNotFoundError(f"VLLM_PYTHON_BIN does not exist: {override}")

    default = Path(
        os.environ.get(
            "VLLM_VENV",
            "/work/nvme/bhph/zluo8/credit_assignment/venvs/vllm-0.25.1-cu129",
        )
    )
    python_bin = default / "bin" / "python"
    if python_bin.is_file():
        return str(python_bin)
    raise FileNotFoundError(
        "expected vLLM python executable at "
        f"{python_bin}; set VLLM_PYTHON_BIN or create the vLLM environment first"
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


def _optional_float_env(name: str) -> str | None:
    raw_value = os.environ.get(name, "").strip()
    if not raw_value:
        return None
    value = float(raw_value)
    if not 0.0 < value < 1.0:
        raise ValueError(f"{name} must be between 0 and 1, got {raw_value}")
    return raw_value


def _vllm_supported_max_lora_rank(adapter_rank: int) -> int:
    supported_ranks = (1, 8, 16, 32, 64, 128, 256, 320, 512)
    for supported_rank in supported_ranks:
        if adapter_rank <= supported_rank:
            return supported_rank
    raise ValueError(
        f"LoRA rank {adapter_rank} exceeds maximum vLLM supported rank "
        f"{supported_ranks[-1]}"
    )


def _optional_positive_int_env(name: str) -> str | None:
    raw_value = os.environ.get(name, "").strip()
    if not raw_value:
        return None
    value = int(raw_value)
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {raw_value}")
    return raw_value


def _positive_int_env_or_default(name: str, default: int) -> int:
    raw_value = _optional_positive_int_env(name)
    if raw_value is None:
        return default
    return int(raw_value)


def _truthy_env(name: str) -> bool:
    return os.environ.get(name, "").strip() in ("1", "true", "True")


def _config_expects_lora(config_nickname: str) -> bool:
    return "lora" in config_nickname.strip().lower()


def _vllm_should_skip_tokenizer_init(
    model_cli_name: str, hf_model_name: str, model_path: str
) -> bool:
    model_cli_name_lower = model_cli_name.strip().lower()
    hf_model_name_lower = hf_model_name.strip().lower()
    model_path_lower = model_path.strip().lower()
    if model_cli_name_lower == "gemma":
        return False
    if "gemma-3" in hf_model_name_lower or "gemma-3" in model_path_lower:
        return False
    return True


def _resolve_peft_adapter_base_model_path(model_path: str) -> str | None:
    model_dir = Path(model_path).expanduser().resolve()
    adapter_config_path = model_dir / "adapter_config.json"
    if not adapter_config_path.exists():
        return None

    base_model_path_file = model_dir / "base_model_path.txt"
    if base_model_path_file.exists():
        base_model_path = base_model_path_file.read_text().strip()
        if base_model_path:
            return base_model_path

    adapter_config = json.loads(adapter_config_path.read_text())
    base_model_path = str(adapter_config.get("base_model_name_or_path", "")).strip()
    if not base_model_path:
        raise ValueError(
            f"PEFT adapter at {model_dir} is missing base_model_name_or_path"
        )
    return base_model_path


def _resolve_peft_adapter_rank(model_path: str) -> int | None:
    adapter_config_path = Path(model_path).expanduser().resolve() / "adapter_config.json"
    if not adapter_config_path.exists():
        return None
    adapter_config = json.loads(adapter_config_path.read_text())
    rank = adapter_config.get("r")
    if rank is None:
        return None
    rank_int = int(rank)
    if rank_int <= 0:
        raise ValueError(f"PEFT adapter rank must be positive, got {rank!r}")
    return rank_int


def _int_list(value: Any, field_name: str) -> list[int]:
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list")
    result: list[int] = []
    for item in value:
        if not isinstance(item, int):
            raise ValueError(f"{field_name} entries must be integers, got {item!r}")
        result.append(item)
    return result


def _parse_token_id_key(key: Any) -> int | None:
    if isinstance(key, int):
        return key
    if not isinstance(key, str):
        return None
    stripped = key.strip()
    if stripped.startswith("token_id:"):
        stripped = stripped.removeprefix("token_id:")
    try:
        return int(stripped)
    except ValueError:
        return None


def _translate_sglang_generate_to_vllm(
    payload: dict[str, Any], served_model_name: str
) -> dict[str, Any]:
    sampling_params = payload.get("sampling_params")
    if not isinstance(sampling_params, dict):
        sampling_params = {}

    request: dict[str, Any] = {
        "model": served_model_name,
        "prompt": _int_list(payload.get("input_ids"), "input_ids"),
        "max_tokens": int(sampling_params.get("max_new_tokens", 16)),
        "temperature": float(sampling_params.get("temperature", 0.0)),
        "return_token_ids": True,
        "return_tokens_as_token_ids": True,
    }
    if payload.get("return_logprob"):
        request["logprobs"] = int(payload.get("top_logprobs_num", 8))
    if "sampling_seed" in sampling_params:
        request["seed"] = int(sampling_params["sampling_seed"])
    if "stop" in sampling_params:
        request["stop"] = sampling_params["stop"]
        if sampling_params.get("no_stop_trim"):
            request["include_stop_str_in_output"] = True
    if "top_p" in sampling_params:
        request["top_p"] = float(sampling_params["top_p"])
    if "top_k" in sampling_params:
        request["top_k"] = int(sampling_params["top_k"])
    return request


def _translate_vllm_completion_to_sglang(response: dict[str, Any]) -> dict[str, Any]:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        return {"error": {"message": f"vLLM response missing choices: {response}"}}
    choice = choices[0]
    if not isinstance(choice, dict):
        return {"error": {"message": f"vLLM choice must be an object: {choice!r}"}}

    token_ids = _int_list(choice.get("token_ids", []), "choices[0].token_ids")
    result: dict[str, Any] = {
        "text": choice.get("text", ""),
        "output_ids": token_ids,
    }

    logprobs = choice.get("logprobs")
    if isinstance(logprobs, dict):
        token_logprobs_raw = logprobs.get("token_logprobs")
        top_logprobs_raw = logprobs.get("top_logprobs")
        if isinstance(token_logprobs_raw, list):
            output_token_logprobs: list[list[Any]] = []
            for idx, token_id in enumerate(token_ids):
                raw_logprob = (
                    token_logprobs_raw[idx]
                    if idx < len(token_logprobs_raw)
                    else None
                )
                logprob = (
                    float(raw_logprob)
                    if isinstance(raw_logprob, (int, float))
                    else float("-inf")
                )
                output_token_logprobs.append([logprob, token_id, None])

            output_top_logprobs: list[list[list[Any]]] = []
            if isinstance(top_logprobs_raw, list):
                for idx, token_id in enumerate(token_ids):
                    candidates: list[list[Any]] = []
                    raw_candidates = (
                        top_logprobs_raw[idx]
                        if idx < len(top_logprobs_raw)
                        else None
                    )
                    if isinstance(raw_candidates, dict):
                        for key, raw_logprob in raw_candidates.items():
                            candidate_token_id = _parse_token_id_key(key)
                            if candidate_token_id is None:
                                continue
                            logprob = (
                                float(raw_logprob)
                                if isinstance(raw_logprob, (int, float))
                                else float("-inf")
                            )
                            candidates.append([logprob, candidate_token_id, None])
                    if not any(candidate[1] == token_id for candidate in candidates):
                        generated_logprob = output_token_logprobs[idx][0]
                        candidates.append([generated_logprob, token_id, None])
                    candidates.sort(key=lambda candidate: candidate[0], reverse=True)
                    output_top_logprobs.append(candidates[:8])
            result["meta_info"] = {
                "output_token_logprobs": output_token_logprobs,
                "output_top_logprobs": output_top_logprobs,
            }
    return result


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
    backend_name: str,
    model_cli_name: str,
    config_nickname: str,
    epoch: int,
    model_path: str,
    pid: int,
) -> None:
    payload = {
        "backend_name": backend_name,
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


class SglangBackend:
    backend_name = "sglang"

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
        self._upstream_port = upstream_port
        self._wrapper_log_path = Path(wrapper_log_path).expanduser().resolve()
        self._wrapper_log_path.parent.mkdir(parents=True, exist_ok=True)
        if not model_path_obj.exists() and epoch == 0:
            _emit_status(
                "hpc",
                "starting",
                f"initial model missing at {model_path_obj}; downloading {hf_model_name}",
            )
            if _CONN is not None:
                _CONN.send_info(f"Downloading initial model for inference: {hf_model_name}")
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
                "backend_name": self.backend_name,
                "model_cli_name": model_cli_name,
                "config_nickname": config_nickname,
                "epoch": epoch,
            }
            observed = {
                "backend_name": str(existing_state.get("backend_name", "sglang")),
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
            if _CONN is not None:
                _CONN.send_info(
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
        if _CONN is not None:
            _CONN.send_info(
                f"Launching local sglang on port {upstream_port} ({self._upstream_url})"
            )
        with open(
            self._wrapper_log_path, "a", buffering=1, encoding="utf-8"
        ) as log_handle:
            launch_command = [
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
            ]
            mem_fraction_static = _optional_float_env("SGLANG_MEM_FRACTION_STATIC")
            if mem_fraction_static is not None:
                launch_command.extend(["--mem-fraction-static", mem_fraction_static])
            max_running_requests = _optional_positive_int_env("SGLANG_MAX_RUNNING_REQUESTS")
            if max_running_requests is not None:
                launch_command.extend(["--max-running-requests", max_running_requests])
            max_total_tokens = _optional_positive_int_env("SGLANG_MAX_TOTAL_TOKENS")
            if max_total_tokens is not None:
                launch_command.extend(["--max-total-tokens", max_total_tokens])
            self._process = subprocess.Popen(
                launch_command,
                stdout=log_handle,
                stderr=log_handle,
                env=child_env,
            )
        _emit_status(
            "hpc", "starting", f"launching local sglang on port {upstream_port}"
        )
        _wait_health(f"{self._upstream_url}/health", timeout_secs=900)
        if _CONN is not None:
            _CONN.send_info(f"SGLang server is up at {self._upstream_url}")
        _write_hpc_server_state(
            upstream_port=upstream_port,
            backend_name=self.backend_name,
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

    def update_model(self, new_model_path: str) -> dict[str, Any]:
        exit_error = self.unexpected_exit_error()
        if exit_error is not None:
            raise RuntimeError(exit_error)
        return _post_json(
            f"{self._upstream_url}/update_weights_from_disk",
            {"model_path": new_model_path},
        )

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


class VllmBackend:
    backend_name = "vllm"

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
        self._upstream_port = upstream_port
        self._served_model_name = os.environ.get(
            "VLLM_SERVED_MODEL_NAME", "credit-assignment"
        )
        self._base_served_model_name = f"{self._served_model_name}-base"
        self._active_served_model_name = self._served_model_name
        self._runtime_lora_enabled = _config_expects_lora(config_nickname) or _truthy_env(
            "VLLM_ALLOW_RUNTIME_LORA_UPDATING"
        )
        self._loaded_lora_name: str | None = None
        self._wrapper_log_path = Path(wrapper_log_path).expanduser().resolve()
        self._wrapper_log_path.parent.mkdir(parents=True, exist_ok=True)
        self._num_gpus = num_gpus
        self._model_cli_name = model_cli_name
        self._config_nickname = config_nickname
        self._epoch = epoch
        self._hf_model_name = hf_model_name
        self._restart_lock = threading.Lock()
        self._restart_in_progress = False
        self._model_update_health_timeout_secs = _positive_int_env_or_default(
            "VLLM_MODEL_UPDATE_HEALTH_TIMEOUT_SECS", 600
        )

        if not model_path_obj.exists() and epoch == 0:
            _emit_status(
                "hpc",
                "starting",
                f"initial model missing at {model_path_obj}; downloading {hf_model_name}",
            )
            if _CONN is not None:
                _CONN.send_info(f"Downloading initial model for inference: {hf_model_name}")
            ensure_model_snapshot(model_path_obj.parent, hf_model_name)
        if not model_path_obj.exists():
            raise FileNotFoundError(f"model path does not exist: {model_path}")

        self._launch_model(model_path, health_timeout_secs=900)

    def _launch_model(self, model_path: str, *, health_timeout_secs: int) -> None:
        vllm_python = _resolve_vllm_python_executable()
        adapter_base_model_path = _resolve_peft_adapter_base_model_path(model_path)
        vllm_model_path = adapter_base_model_path or model_path
        is_peft_adapter = adapter_base_model_path is not None
        adapter_rank = _resolve_peft_adapter_rank(model_path) if is_peft_adapter else None
        self._current_base_model_path = str(Path(vllm_model_path).expanduser().resolve())
        lora_enabled = self._runtime_lora_enabled or is_peft_adapter
        base_served_model_name = (
            self._base_served_model_name if lora_enabled else self._served_model_name
        )
        self._active_served_model_name = (
            self._served_model_name if is_peft_adapter else base_served_model_name
        )
        child_env = os.environ.copy()
        child_env.setdefault("PYTHONUNBUFFERED", "1")
        child_env.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        if lora_enabled:
            child_env["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "True"
        if os.environ.get("VLLM_INHERIT_LD_LIBRARY_PATH", "").strip() not in (
            "1",
            "true",
            "True",
        ):
            child_env.pop("LD_LIBRARY_PATH", None)

        if _CONN is not None:
            _CONN.send_info(
                f"Launching local vLLM on port {self._upstream_port} ({self._upstream_url})"
            )
        with open(
            self._wrapper_log_path, "a", buffering=1, encoding="utf-8"
        ) as log_handle:
            launch_command = [
                vllm_python,
                "-m",
                "vllm.entrypoints.openai.api_server",
                "--model",
                vllm_model_path,
                "--served-model-name",
                base_served_model_name,
                "--host",
                "127.0.0.1",
                "--port",
                str(self._upstream_port),
                "--dtype",
                os.environ.get("VLLM_DTYPE", "auto"),
            ]
            if _vllm_should_skip_tokenizer_init(
                self._model_cli_name, self._hf_model_name, vllm_model_path
            ):
                launch_command.append("--skip-tokenizer-init")
            if is_peft_adapter:
                launch_command.extend(
                    [
                        "--enable-lora",
                        "--max-loras",
                        "1",
                        "--lora-modules",
                        f"{self._served_model_name}={model_path}",
                    ]
                )
                if adapter_rank is not None:
                    launch_command.extend(
                        ["--max-lora-rank", str(_vllm_supported_max_lora_rank(adapter_rank))]
                    )
            elif lora_enabled:
                max_lora_rank = _optional_positive_int_env("VLLM_MAX_LORA_RANK")
                launch_command.extend(
                    [
                        "--enable-lora",
                        "--max-loras",
                        "1",
                        "--max-lora-rank",
                        max_lora_rank or "64",
                    ]
                )
            if self._num_gpus > 1:
                launch_command.extend(["--tensor-parallel-size", str(self._num_gpus)])
            gpu_memory_utilization = _optional_float_env("VLLM_GPU_MEMORY_UTILIZATION")
            if gpu_memory_utilization is not None:
                launch_command.extend(["--gpu-memory-utilization", gpu_memory_utilization])
            max_model_len = _optional_positive_int_env("VLLM_MAX_MODEL_LEN")
            if max_model_len is not None:
                launch_command.extend(["--max-model-len", max_model_len])
            max_num_seqs = _optional_positive_int_env("VLLM_MAX_NUM_SEQS")
            if max_num_seqs is not None:
                launch_command.extend(["--max-num-seqs", max_num_seqs])
            max_num_batched_tokens = _optional_positive_int_env(
                "VLLM_MAX_NUM_BATCHED_TOKENS"
            )
            if max_num_batched_tokens is not None:
                launch_command.extend(["--max-num-batched-tokens", max_num_batched_tokens])
            if os.environ.get("VLLM_ENFORCE_EAGER", "").strip() in ("1", "true", "True"):
                launch_command.append("--enforce-eager")

            _emit_status(
                "hpc",
                "starting",
                "vLLM launch command: " + " ".join(launch_command),
            )
            self._process = subprocess.Popen(
                launch_command,
                stdout=log_handle,
                stderr=log_handle,
                env=child_env,
            )
        _emit_status(
            "hpc", "starting", f"launching local vLLM on port {self._upstream_port}"
        )
        if self._process is None:
            raise RuntimeError("internal error: vLLM process was not created")
        _wait_health_or_process_exit(
            f"{self._upstream_url}/health",
            self._process,
            timeout_secs=health_timeout_secs,
            log_path=self._wrapper_log_path,
        )
        if _CONN is not None:
            _CONN.send_info(f"vLLM server is up at {self._upstream_url}")
        _write_hpc_server_state(
            upstream_port=self._upstream_port,
            backend_name=self.backend_name,
            model_cli_name=self._model_cli_name,
            config_nickname=self._config_nickname,
            epoch=self._epoch,
            model_path=model_path,
            pid=self._process.pid if self._process is not None else -1,
        )
        _emit_status("hpc", "ready", "local vLLM health check passed")

    def health(self) -> dict[str, Any]:
        exit_error = self.unexpected_exit_error()
        if exit_error is not None:
            raise RuntimeError(exit_error)
        return _probe_health(f"{self._upstream_url}/health")

    def generate(self, payload: dict[str, Any]) -> dict[str, Any]:
        exit_error = self.unexpected_exit_error()
        if exit_error is not None:
            raise RuntimeError(exit_error)
        vllm_payload = _translate_sglang_generate_to_vllm(
            payload, self._active_served_model_name
        )
        response = _post_json(f"{self._upstream_url}/v1/completions", vllm_payload)
        return _translate_vllm_completion_to_sglang(response)

    def _load_lora_adapter_runtime(self, new_model_path: str) -> dict[str, Any]:
        adapter_base_model_path = _resolve_peft_adapter_base_model_path(new_model_path)
        if adapter_base_model_path is None:
            raise ValueError(f"model path is not a PEFT adapter: {new_model_path}")
        if Path(adapter_base_model_path).expanduser().resolve() != Path(
            self._current_base_model_path
        ).expanduser().resolve():
            raise ValueError(
                "runtime LoRA update requires the adapter base model to match "
                f"the running vLLM base model; running={self._current_base_model_path}, "
                f"adapter_base={adapter_base_model_path}"
            )
        if self._loaded_lora_name is not None:
            try:
                _post_json_accept_any_response(
                    f"{self._upstream_url}/v1/unload_lora_adapter",
                    {"lora_name": self._loaded_lora_name},
                    timeout=120.0,
                )
            except HTTPError as error:
                if error.code != 404:
                    raise
        result = _post_json_accept_any_response(
            f"{self._upstream_url}/v1/load_lora_adapter",
            {"lora_name": self._served_model_name, "lora_path": new_model_path},
            timeout=float(self._model_update_health_timeout_secs),
        )
        self._loaded_lora_name = self._served_model_name
        self._active_served_model_name = self._served_model_name
        self._epoch += 1
        _emit_status(
            "hpc",
            "ready",
            f"runtime-loaded vLLM LoRA adapter from {new_model_path}",
        )
        return result

    def update_model(self, new_model_path: str) -> dict[str, Any]:
        exit_error = self.unexpected_exit_error()
        if exit_error is not None:
            raise RuntimeError(exit_error)
        with self._restart_lock:
            self._restart_in_progress = True
        try:
            if self._runtime_lora_enabled and _resolve_peft_adapter_base_model_path(
                new_model_path
            ):
                return self._load_lora_adapter_runtime(new_model_path)
            if self._process is not None and self._process.poll() is None:
                _emit_status(
                    "hpc",
                    "starting",
                    f"stopping vLLM pid={self._process.pid} before model update",
                )
                self._process.terminate()
                try:
                    self._process.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    _emit_status(
                        "hpc",
                        "starting",
                        f"vLLM pid={self._process.pid} did not terminate; killing",
                    )
                    self._process.kill()
                    self._process.wait(timeout=30)
                _emit_status(
                    "hpc",
                    "starting",
                    "old vLLM process stopped for planned model update "
                    f"(return_code={self._process.returncode})",
                )
            self._epoch += 1
            self._launch_model(
                new_model_path,
                health_timeout_secs=self._model_update_health_timeout_secs,
            )
        except Exception as error:
            message = (
                f"failed to restart vLLM for model update to {new_model_path}: {error}"
            )
            _emit_event(
                {
                    "type": "error",
                    "backend": self.backend_name,
                    "error_code": "VLLM_MODEL_UPDATE_RELAUNCH_FAILED",
                    "error_message": message,
                    "timestamp": time.time(),
                }
            )
            raise RuntimeError(message) from error
        finally:
            with self._restart_lock:
                self._restart_in_progress = False
        return {"status": "ok", "message": "vLLM model restarted"}

    def unexpected_exit_error(self) -> str | None:
        if self._process is None:
            return None
        with self._restart_lock:
            if self._restart_in_progress:
                return None
        return_code = self._process.poll()
        if return_code is None:
            return None
        return (
            f"local vLLM process exited unexpectedly with return code {return_code}; "
            f"last log lines:\n{_tail_text(self._wrapper_log_path)}"
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


def _select_backend_class(backend: str) -> type[SglangBackend] | type[VllmBackend]:
    backend = backend.strip().lower()
    if backend in ("", "sglang"):
        return SglangBackend
    if backend == "vllm":
        return VllmBackend
    raise ValueError("INFERENCE_BACKEND must be one of: sglang, vllm")


class ThreadedHTTPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True


class InferenceWrapperArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    listen_port: int
    inference_backend: str
    num_gpus: int
    epoch: int
    model_cli_name: str
    config_nickname: str
    hf_model_name: str
    model_path: str
    wrapper_log_path: str


def main() -> int:
    global _CONN

    _set_process_name("inference_wrapper")
    backend_name = "unknown"
    conn: RustParentConnection[Any] | None = None
    try:
        conn = RustParentConnection(
            read_orchestrator_socket_path(),
            stdin_model=InferenceWrapperArgs,
        )
        _CONN = conn
        stdin_data = conn.stdin_data
        _configure_wrapper_log_file(stdin_data.wrapper_log_path)
        conn.send_info("Inference wrapper started")
        _emit_inference_text_identity(
            stdin_data.model_cli_name,
            stdin_data.config_nickname,
            stdin_data.epoch,
            stdin_data.hf_model_name,
            stdin_data.listen_port,
        )
        if stdin_data.listen_port <= 0:
            raise ValueError("--listen-port must be positive")

        backend_class = _select_backend_class(stdin_data.inference_backend)
        backend_name = backend_class.backend_name
        backend = backend_class(
            stdin_data.model_path,
            stdin_data.num_gpus,
            stdin_data.listen_port + 1,
            stdin_data.epoch,
            stdin_data.hf_model_name,
            stdin_data.model_cli_name,
            stdin_data.config_nickname,
            stdin_data.wrapper_log_path,
        )

        _emit_inference_identity(
            backend_name,
            stdin_data.hf_model_name,
            stdin_data.config_nickname,
            stdin_data.epoch,
        )

        _emit_status(
            backend_name,
            "starting",
            f"binding local wrapper server on port {stdin_data.listen_port}",
        )
        conn.send_info(f"Binding inference wrapper server on port {stdin_data.listen_port}")

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
                if self.path not in ("/generate", "/update_model"):
                    self.send_response(404)
                    self.end_headers()
                    return
                content_len = int(self.headers.get("content-length", "0"))
                raw = self.rfile.read(content_len)
                try:
                    payload = json.loads(raw.decode("utf-8"))
                    if self.path == "/update_model":
                        new_model_path = payload["model_path"]
                        result = backend.update_model(new_model_path)
                    else:
                        result = backend.generate(payload)
                    body = json.dumps(result).encode("utf-8")
                    self.send_response(200)
                    self.send_header("content-type", "application/json")
                    self.send_header("content-length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                except Exception as error:  # noqa: BLE001
                    error_code = (
                        "INFERENCE_UPDATE_MODEL_FAILED"
                        if self.path == "/update_model"
                        else "INFERENCE_GENERATE_FAILED"
                    )
                    _emit_event(
                        {
                            "type": "error",
                            "backend": backend_name,
                            "error_code": error_code,
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

        server = ThreadedHTTPServer(("127.0.0.1", stdin_data.listen_port), Handler)
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
                            "error_code": "INFERENCE_BACKEND_PROCESS_EXITED",
                            "error_message": exit_error,
                            "timestamp": time.time(),
                        }
                    )
                    conn.send_error(f"Inference backend exited unexpectedly: {exit_error}")
                    shutdown_event.set()
                    threading.Thread(target=server.shutdown, daemon=True).start()
                    return
                shutdown_event.wait(timeout=1.0)

        threading.Thread(target=_watch_backend_process, daemon=True).start()
        _emit_status(backend_name, "ready", "inference wrapper server is ready")
        conn.send_info("Inference wrapper server is ready")

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
                conn.send_error(
                    f"Inference wrapper stopping after backend failure: {error_message}"
                )
                _emit_event(
                    {
                        "type": "result",
                        "backend": backend_name,
                        "ok": False,
                        "error_code": "INFERENCE_BACKEND_PROCESS_EXITED",
                        "error_message": error_message,
                        "message": error_message,
                        "timestamp": time.time(),
                    }
                )
                return 1
            conn.send_info("Inference wrapper stopped")
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
        if conn is not None:
            conn.send_error(f"Inference wrapper failed to start: {error}")
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
