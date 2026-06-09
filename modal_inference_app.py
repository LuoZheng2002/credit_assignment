import os
import subprocess
import threading
import time
from typing import Any

import modal
import requests

MINUTES = 60
APP_NAME = "credit-assignment-inference-service"
SGLANG_PORT = 30000
REGION = os.environ.get("MODAL_REGION", "us-east")
GPU = "H100:1"
MODEL_NAME = os.environ.get("SGLANG_MODEL_NAME", "Qwen/Qwen3.5-4B")

HF_CACHE_PATH = "/root/.cache/huggingface"

service_state_volume = modal.Volume.from_name(
    "credit-assignment-modal-service-state", create_if_missing=True
)
hf_cache_volume = modal.Volume.from_name("credit-assignment-hf-cache", create_if_missing=True)

base_image = (
    modal.Image.from_registry("lmsysorg/sglang:latest")
    .entrypoint([])
    .pip_install("requests>=2.32.0")
    .add_local_dir(".", remote_path="/root/credit_assignment")
    .env(
        {
            "HF_HUB_CACHE": HF_CACHE_PATH,
            "PYTHONPATH": "/root/credit_assignment",
        }
    )
)

app = modal.App(name=APP_NAME)

_MODAL_INFERENCE_LOCK = threading.Lock()
_MODAL_INFERENCE_PROCESS: subprocess.Popen[bytes] | None = None
_MODAL_INFERENCE_MODEL_NAME: str | None = None


def _wait_for_local_health(port: int, timeout_secs: int = 600) -> None:
    deadline = time.time() + timeout_secs
    while time.time() < deadline:
        try:
            response = requests.get(f"http://127.0.0.1:{port}/health", timeout=2)
            if response.status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(1)
    raise TimeoutError(f"timed out waiting for local server health on port {port}")


def _ensure_modal_inference_sglang(model_name: str) -> None:
    global _MODAL_INFERENCE_PROCESS
    global _MODAL_INFERENCE_MODEL_NAME
    with _MODAL_INFERENCE_LOCK:
        if _MODAL_INFERENCE_PROCESS is not None and _MODAL_INFERENCE_PROCESS.poll() is None:
            if _MODAL_INFERENCE_MODEL_NAME == model_name:
                return
            _MODAL_INFERENCE_PROCESS.terminate()
            _MODAL_INFERENCE_PROCESS.wait(timeout=30)
        _MODAL_INFERENCE_PROCESS = subprocess.Popen(
            [
                "python",
                "-m",
                "sglang.launch_server",
                "--model-path",
                model_name,
                "--served-model-name",
                model_name,
                "--host",
                "127.0.0.1",
                "--port",
                str(SGLANG_PORT),
                "--tp",
                "1",
            ]
        )
        _MODAL_INFERENCE_MODEL_NAME = model_name
    _wait_for_local_health(SGLANG_PORT)


@app.cls(
    image=base_image,
    gpu=GPU,
    region=REGION,
    startup_timeout=20 * MINUTES,
    min_containers=0,
    max_containers=1,
    timeout=20 * MINUTES,
    volumes={
        "/mnt/service-state": service_state_volume,
        HF_CACHE_PATH: hf_cache_volume,
    },
)
class ExperimentService:
    model_cli_name: str = modal.parameter()
    config_nickname: str = modal.parameter()
    model_name: str = modal.parameter(default=MODEL_NAME)

    @modal.method()
    def generate(self, payload: dict[str, Any]) -> dict[str, Any]:
        _ensure_modal_inference_sglang(self.model_name)
        response = requests.post(
            f"http://127.0.0.1:{SGLANG_PORT}/generate",
            json=payload,
            timeout=600,
        )
        response.raise_for_status()
        return response.json()


@app.local_entrypoint()
def show_url() -> None:
    print("Deploy with: modal deploy modal_inference_app.py")
