import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download
import modal
import requests

MINUTES = 60
APP_NAME = "credit-assignment-inference-service"
SGLANG_PORT = 30000
REGION = "us-west"
GPU = "H100:1"
DEFAULT_ARTIFACT_ROOT_DIR = "/mnt/service-state"

HF_CACHE_PATH = "/mnt/hf-cache"

service_state_volume = modal.Volume.from_name(
    "credit-assignment-modal-service-state", create_if_missing=True
)
hf_cache_volume = modal.Volume.from_name("credit-assignment-hf-cache", create_if_missing=True)

base_image = (
    modal.Image.from_registry("lmsysorg/sglang:latest")
    .entrypoint([])
    .pip_install("requests>=2.32.0", "huggingface_hub>=0.35.0")
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
_MODAL_INFERENCE_MODEL_PATH: str | None = None


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


def _model_parent_dir(
    artifact_root_dir: str,
    model_cli_name: str,
    config_nickname: str,
    epoch: int,
) -> Path:
    root = Path(artifact_root_dir)
    if epoch == 0:
        return root / "results" / model_cli_name
    return root / "results" / model_cli_name / config_nickname / f"epoch_{epoch}"


def _ensure_initial_model_if_needed(
    artifact_root_dir: str,
    model_cli_name: str,
    config_nickname: str,
    epoch: int,
    hf_model_name: str,
) -> str:
    parent_dir = _model_parent_dir(artifact_root_dir, model_cli_name, config_nickname, epoch)
    model_dir = parent_dir / "model"
    if model_dir.exists():
        return str(model_dir)
    if epoch != 0:
        raise FileNotFoundError(
            f"model directory missing for non-initial epoch at {model_dir}; expected trained model"
        )
    model_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=hf_model_name, local_dir=str(model_dir), token=os.environ.get("HF_TOKEN") or None)
    return str(model_dir)


def _ensure_modal_inference_sglang(model_path: str) -> None:
    global _MODAL_INFERENCE_PROCESS
    global _MODAL_INFERENCE_MODEL_PATH
    with _MODAL_INFERENCE_LOCK:
        if _MODAL_INFERENCE_PROCESS is not None and _MODAL_INFERENCE_PROCESS.poll() is None:
            if _MODAL_INFERENCE_MODEL_PATH == model_path:
                return
            _MODAL_INFERENCE_PROCESS.terminate()
            _MODAL_INFERENCE_PROCESS.wait(timeout=30)
        _MODAL_INFERENCE_PROCESS = subprocess.Popen(
            [
                "python",
                "-m",
                "sglang.launch_server",
                "--model-path",
                model_path,
                "--served-model-name",
                model_path,
                "--host",
                "127.0.0.1",
                "--port",
                str(SGLANG_PORT),
                "--tp",
                "1",
            ]
        )
        _MODAL_INFERENCE_MODEL_PATH = model_path
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
    model_name: str = modal.parameter()
    epoch: int = modal.parameter(default=0)
    artifact_root_dir: str = modal.parameter(default=DEFAULT_ARTIFACT_ROOT_DIR)

    def _resolve_model_path(self) -> str:
        return _ensure_initial_model_if_needed(
            artifact_root_dir=self.artifact_root_dir,
            model_cli_name=self.model_cli_name,
            config_nickname=self.config_nickname,
            epoch=int(self.epoch),
            hf_model_name=self.model_name,
        )

    @modal.method()
    def health(self) -> dict[str, str]:
        _ensure_modal_inference_sglang(self._resolve_model_path())
        return {"status": "ok"}

    @modal.method()
    def generate(self, payload: dict[str, Any]) -> dict[str, Any]:
        _ensure_modal_inference_sglang(self._resolve_model_path())
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
