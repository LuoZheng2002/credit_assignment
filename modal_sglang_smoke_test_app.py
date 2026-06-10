from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import modal

from src_py.load_model_to_path import ensure_model_snapshot

MINUTES = 60
APP_NAME = "credit-assignment-sglang-smoke-test"
REGION = "us-west"
SGLANG_PORT = 30000
HF_CACHE_PATH = "/mnt/hf-cache"
DEFAULT_DOWNLOAD_PARENT_DIR = "/mnt/service-state/smoke-test"

service_state_volume = modal.Volume.from_name(
    "credit-assignment-modal-service-state", create_if_missing=True
)
hf_cache_volume = modal.Volume.from_name(
    "credit-assignment-hf-cache", create_if_missing=True
)

smoke_test_image = modal.Image.from_dockerfile(
    "Dockerfile.modal-mirror",
    ignore=modal.FilePatternMatcher.from_file(
        str(Path(__file__).with_name(".dockerignore"))
    ),
).env(
    {
        "HF_HUB_CACHE": HF_CACHE_PATH,
        "PYTHONPATH": "/workspace",
    }
)

app = modal.App(name=APP_NAME)


def _resolve_sglang_python_executable() -> str:
    python_bin = Path("/workspace/pyprojects/sglang/.venv/bin/python")
    if python_bin.is_file():
        return str(python_bin)
    raise FileNotFoundError(
        "expected sglang python executable at "
        f"{python_bin}; check Dockerfile.modal-mirror image build"
    )


def _collect_sglang_package_versions(sglang_python: str) -> dict[str, str]:
    freeze_cmd = ["uv", "pip", "freeze", "--python", sglang_python]
    result = subprocess.run(
        freeze_cmd,
        cwd="/workspace",
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        install_pip_cmd = ["uv", "pip", "install", "--python", sglang_python, "pip"]
        install_pip = subprocess.run(
            install_pip_cmd,
            cwd="/workspace",
            capture_output=True,
            text=True,
            check=False,
        )
        if install_pip.returncode == 0:
            result = subprocess.run(
                freeze_cmd,
                cwd="/workspace",
                capture_output=True,
                text=True,
                check=False,
            )
    if result.returncode != 0:
        stderr = result.stderr.strip() or result.stdout.strip() or "unknown error"
        return {"error": f"uv pip freeze failed rc={result.returncode}: {stderr}"}

    packages: dict[str, str] = {}
    for line in result.stdout.splitlines():
        normalized = line.strip()
        if not normalized:
            continue
        lower = normalized.lower()
        if lower.startswith("kernels==") or lower.startswith("sglang=="):
            name, _, version = normalized.partition("==")
            packages[name] = version or normalized
    return packages


def _probe_kernels_module(sglang_python: str) -> dict[str, str]:
    probe_cmd = [
        sglang_python,
        "-c",
        "import json, kernels; print(json.dumps({'version': kernels.__version__, 'file': kernels.__file__}))",
    ]
    result = subprocess.run(
        probe_cmd,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip() or result.stdout.strip() or "unknown error"
        return {
            "error": f"kernels import probe failed rc={result.returncode}: {stderr}"
        }
    try:
        parsed = json.loads(result.stdout.strip())
    except json.JSONDecodeError:
        return {"raw": result.stdout.strip()}
    if not isinstance(parsed, dict):
        return {"raw": result.stdout.strip()}
    return {str(key): str(value) for key, value in parsed.items()}


def _post_json(
    url: str, payload: dict[str, Any], timeout_secs: float
) -> tuple[int, str]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"content-type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_secs) as response:
            return response.status, response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode("utf-8", errors="replace")


def _wait_for_completions_ready(
    *,
    port: int,
    served_model_name: str,
    process: subprocess.Popen[bytes],
    timeout_secs: int,
) -> dict[str, Any]:
    deadline = time.time() + timeout_secs
    last_error = ""
    payload = {
        "model": served_model_name,
        "prompt": "ready check",
        "max_tokens": 1,
        "temperature": 0.0,
    }
    while time.time() < deadline:
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(
                f"sglang subprocess exited before /v1/completions became ready; rc={return_code}"
            )
        try:
            status_code, response_body = _post_json(
                f"http://127.0.0.1:{port}/v1/completions",
                payload,
                timeout_secs=15,
            )
            if status_code == 200:
                try:
                    parsed_body = json.loads(response_body)
                except json.JSONDecodeError:
                    parsed_body = {"raw": response_body}
                return {
                    "status_code": status_code,
                    "response": parsed_body,
                }
            last_error = f"status={status_code}, body={response_body[:1000]}"
        except urllib.error.URLError as error:
            last_error = f"url error: {error}"
        except TimeoutError:
            last_error = "request timed out"
        time.sleep(2)
    raise TimeoutError(
        "timed out waiting for /v1/completions readiness "
        f"on port {port}; last_error={last_error}"
    )


def _resolve_model_path(
    *,
    model_path: str,
    model_api_name: str,
    download_parent_dir: str,
) -> str:
    explicit_model_path = model_path.strip()
    hf_model_name = model_api_name.strip()
    if bool(explicit_model_path) == bool(hf_model_name):
        raise ValueError("pass exactly one of model_path or model_api_name")
    if explicit_model_path:
        resolved_path = Path(explicit_model_path)
        if not resolved_path.exists():
            raise FileNotFoundError(
                f"model_path does not exist inside the Modal container: {resolved_path}"
            )
        return str(resolved_path)
    downloaded_model_path = ensure_model_snapshot(
        Path(download_parent_dir), hf_model_name
    )
    return str(downloaded_model_path)


@app.function(
    image=smoke_test_image,
    gpu="H100:1",
    region=REGION,
    startup_timeout=20 * MINUTES,
    timeout=30 * MINUTES,
    volumes={
        "/mnt/service-state": service_state_volume,
        HF_CACHE_PATH: hf_cache_volume,
    },
)
def run_sglang_smoke_test(
    model_path: str = "",
    model_api_name: str = "",
    served_model_name: str = "",
    download_parent_dir: str = DEFAULT_DOWNLOAD_PARENT_DIR,
    timeout_secs: int = 20 * MINUTES,
    port: int = SGLANG_PORT,
) -> dict[str, Any]:
    resolved_model_path = _resolve_model_path(
        model_path=model_path,
        model_api_name=model_api_name,
        download_parent_dir=download_parent_dir,
    )
    effective_served_model_name = (
        served_model_name.strip() or model_api_name.strip() or resolved_model_path
    )
    sglang_python = _resolve_sglang_python_executable()
    package_versions = _collect_sglang_package_versions(sglang_python)
    kernels_probe = _probe_kernels_module(sglang_python)

    process = subprocess.Popen(
        [
            sglang_python,
            "-m",
            "sglang.launch_server",
            "--model-path",
            resolved_model_path,
            "--served-model-name",
            effective_served_model_name,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--dp",
            "1",
        ],
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    try:
        ready_result = _wait_for_completions_ready(
            port=port,
            served_model_name=effective_served_model_name,
            process=process,
            timeout_secs=timeout_secs,
        )
        return {
            "ok": True,
            "sglang_python": sglang_python,
            "package_versions": package_versions,
            "kernels_probe": kernels_probe,
            "model_path": resolved_model_path,
            "served_model_name": effective_served_model_name,
            "readiness": ready_result,
        }
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()


@app.local_entrypoint()
def main(
    model_path: str = "",
    model_api_name: str = "",
    served_model_name: str = "",
    download_parent_dir: str = DEFAULT_DOWNLOAD_PARENT_DIR,
    timeout_secs: int = 20 * MINUTES,
) -> None:
    result = run_sglang_smoke_test.remote(
        model_path=model_path,
        model_api_name=model_api_name,
        served_model_name=served_model_name,
        download_parent_dir=download_parent_dir,
        timeout_secs=timeout_secs,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
