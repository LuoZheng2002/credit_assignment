import os
import subprocess
import time
from pathlib import Path

from huggingface_hub import snapshot_download
import modal
import modal.experimental
import requests

from src_py.modal.inference_deployment_common import (
    load_materialized_deploy_config,
)

MINUTES = 60
APP_NAME = "credit-assignment-inference-service"
SGLANG_PORT = 30000
REGION = "us-west"
DEFAULT_ARTIFACT_ROOT_DIR = "/mnt/service-state"
TARGET_INPUTS = 300
MIN_CONTAINERS = 1

HF_CACHE_PATH = "/mnt/hf-cache"

service_state_volume = modal.Volume.from_name(
    "credit-assignment-modal-service-state", create_if_missing=True
)
hf_cache_volume = modal.Volume.from_name("credit-assignment-hf-cache", create_if_missing=True)

base_image = (
    modal.Image.from_registry("lmsysorg/sglang:latest")
    .entrypoint([])
    .pip_install("requests>=2.32.0", "huggingface_hub>=0.35.0")
    .add_local_dir("src_py", remote_path="/root/credit_assignment/src_py", copy=True)
    .env(
        {
            "HF_HUB_CACHE": HF_CACHE_PATH,
            "PYTHONPATH": "/root/credit_assignment",
        }
    )
)

app = modal.App(name=APP_NAME)
_DEPLOY_CONFIG = load_materialized_deploy_config()
DEPLOY_MODEL_CLI_NAME = str(_DEPLOY_CONFIG["DEPLOY_MODEL_CLI_NAME"])
DEPLOY_MODEL_API_NAME = str(_DEPLOY_CONFIG["DEPLOY_MODEL_API_NAME"])
DEPLOY_CONFIG_NICKNAME = str(_DEPLOY_CONFIG["DEPLOY_CONFIG_NICKNAME"])
DEPLOY_EPOCH = int(_DEPLOY_CONFIG["DEPLOY_EPOCH"])
DEPLOY_NUM_GPUS = int(_DEPLOY_CONFIG["DEPLOY_NUM_GPUS"])
DEPLOY_ARTIFACT_ROOT_DIR = str(_DEPLOY_CONFIG["DEPLOY_ARTIFACT_ROOT_DIR"])
if DEPLOY_NUM_GPUS <= 0:
    raise RuntimeError(f"DEPLOY_NUM_GPUS must be positive, got {DEPLOY_NUM_GPUS}")
GPU = f"H100:{DEPLOY_NUM_GPUS}"


def _wait_for_local_health(port: int, timeout_secs: int = 20 * MINUTES) -> None:
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


def _warmup(port: int) -> None:
    payload = {
        "text": "ready check",
        "sampling_params": {
            "max_new_tokens": 1,
            "temperature": 0.0,
        },
    }
    for _ in range(3):
        response = requests.post(
            f"http://127.0.0.1:{port}/generate",
            json=payload,
            timeout=20,
        )
        response.raise_for_status()


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
    config_path = model_dir / "config.json"
    if model_dir.exists() and config_path.is_file():
        return str(model_dir)
    if epoch != 0:
        raise FileNotFoundError(
            f"model directory missing for non-initial epoch at {model_dir}; expected trained model"
        )
    model_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=hf_model_name, local_dir=str(model_dir), token=os.environ.get("HF_TOKEN") or None)
    return str(model_dir)


@app.cls(
    image=base_image,
    gpu=GPU,
    region=REGION,
    startup_timeout=20 * MINUTES,
    min_containers=MIN_CONTAINERS,
    max_containers=1,
    timeout=20 * MINUTES,
    volumes={
        "/mnt/service-state": service_state_volume,
        HF_CACHE_PATH: hf_cache_volume,
    },
)
@modal.experimental.http_server(
    port=SGLANG_PORT,
    proxy_regions=[REGION],
    exit_grace_period=15,
)
@modal.concurrent(target_inputs=TARGET_INPUTS)
class ExperimentService:
    @modal.enter()
    def startup(self) -> None:
        model_path = _ensure_initial_model_if_needed(
            artifact_root_dir=DEPLOY_ARTIFACT_ROOT_DIR,
            model_cli_name=DEPLOY_MODEL_CLI_NAME,
            config_nickname=DEPLOY_CONFIG_NICKNAME,
            epoch=DEPLOY_EPOCH,
            hf_model_name=DEPLOY_MODEL_API_NAME,
        )
        self._process = subprocess.Popen(
            [
                "python",
                "-m",
                "sglang.launch_server",
                "--model-path",
                model_path,
                "--served-model-name",
                model_path,
                "--host",
                "0.0.0.0",
                "--port",
                str(SGLANG_PORT),
                "--dp",
                str(DEPLOY_NUM_GPUS),
            ]
        )
        _wait_for_local_health(SGLANG_PORT)
        _warmup(SGLANG_PORT)

    @modal.exit()
    def stop(self) -> None:
        process = getattr(self, "_process", None)
        if process is None:
            return
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()

@app.local_entrypoint()
def show_url() -> None:
    print("Deploy with: modal deploy modal_inference_app.py")
