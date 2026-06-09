import signal
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import modal

from src_py.modal.training_deployment_common import (
    load_materialized_deploy_config,
)
from src_py.load_model_to_path import ensure_model_snapshot

MINUTES = 60
APP_NAME = "credit-assignment-training-service"
REGION = "us-west"
_DEPLOY_CONFIG = load_materialized_deploy_config()
DEPLOY_MODEL_CLI_NAME = str(_DEPLOY_CONFIG["DEPLOY_MODEL_CLI_NAME"])
DEPLOY_MODEL_API_NAME = str(_DEPLOY_CONFIG["DEPLOY_MODEL_API_NAME"])
DEPLOY_CONFIG_NICKNAME = str(_DEPLOY_CONFIG["DEPLOY_CONFIG_NICKNAME"])
DEPLOY_EPOCH = int(_DEPLOY_CONFIG["DEPLOY_EPOCH"])
DEPLOY_NUM_GPUS = int(_DEPLOY_CONFIG["DEPLOY_NUM_GPUS"])
DEPLOY_ARTIFACT_ROOT_DIR = Path(str(_DEPLOY_CONFIG["DEPLOY_ARTIFACT_ROOT_DIR"]))
if DEPLOY_NUM_GPUS <= 0:
    raise RuntimeError(f"DEPLOY_NUM_GPUS must be positive, got {DEPLOY_NUM_GPUS}")
GPU = f"H100:{DEPLOY_NUM_GPUS}"

HF_CACHE_PATH = "/mnt/hf-cache"

service_state_volume = modal.Volume.from_name(
    "credit-assignment-modal-service-state", create_if_missing=True
)
hf_cache_volume = modal.Volume.from_name("credit-assignment-hf-cache", create_if_missing=True)

training_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("torch", "transformers", "accelerate", "datasets>=4.8.5")
    .uv_pip_install("peft", "flash-linear-attention")
    .uv_pip_install("numpy>=2.4.6", "scipy>=1.17.1", "numexpr>=2.14.1")
    .uv_pip_install("python-dotenv>=1.0.0", "matplotlib>=3.10.9", "sympy>=1.14.0")
    .uv_pip_install("huggingface_hub")
    .add_local_dir("src_py", remote_path="/root/credit_assignment/src_py", copy=True)
    .env(
        {
            "HF_HUB_CACHE": HF_CACHE_PATH,
            "PYTHONPATH": "/root/credit_assignment",
        }
    )
)

app = modal.App(name=APP_NAME)


def _to_toml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value == float("inf") or value == float("-inf"):
            raise ValueError("inf is not supported in train_request.toml")
        if value != value:
            raise ValueError("nan is not supported in train_request.toml")
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
            raise ValueError(f"Nested training_config field is not supported for key: {key}")
        lines.append(f"{key} = {_to_toml_scalar(value)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _model_parent_dir(
    artifact_root_dir: Path,
    model_cli_name: str,
    config_nickname: str,
    epoch: int,
) -> Path:
    if epoch == 0:
        return artifact_root_dir / "results" / model_cli_name
    return artifact_root_dir / "results" / model_cli_name / config_nickname / f"epoch_{epoch}"


def _ensure_initial_model_if_needed(
    artifact_root_dir: Path,
    model_cli_name: str,
    config_nickname: str,
    epoch: int,
    hf_model_name: str,
) -> Path:
    parent_dir = _model_parent_dir(artifact_root_dir, model_cli_name, config_nickname, epoch)
    model_dir = parent_dir / "model"
    config_path = model_dir / "config.json"
    if model_dir.exists() and config_path.is_file():
        return model_dir
    if epoch != 0:
        raise FileNotFoundError(
            f"model directory missing for non-initial epoch at {model_dir}; expected trained model"
        )
    ensure_model_snapshot(parent_dir, hf_model_name)
    return model_dir


def _run_training_subprocess(
    training_config: dict[str, Any],
    trajectory_bytes: bytes,
    num_gpus: int,
) -> dict[str, Any]:
    if num_gpus <= 0:
        return {
            "ok": False,
            "error_code": "INVALID_NUM_GPUS",
            "error": f"num_gpus must be positive, got {num_gpus}",
        }
    if num_gpus != DEPLOY_NUM_GPUS:
        return {
            "ok": False,
            "error_code": "MODAL_GPU_COUNT_MISMATCH",
            "error": (
                f"requested num_gpus={num_gpus} but deployed container has "
                f"DEPLOY_NUM_GPUS={DEPLOY_NUM_GPUS}; redeploy modal_training_app.py "
                "with matching wrapper deploy config"
            ),
        }

    _ensure_initial_model_if_needed(
        artifact_root_dir=DEPLOY_ARTIFACT_ROOT_DIR,
        model_cli_name=DEPLOY_MODEL_CLI_NAME,
        config_nickname=DEPLOY_CONFIG_NICKNAME,
        epoch=DEPLOY_EPOCH,
        hf_model_name=DEPLOY_MODEL_API_NAME,
    )

    with tempfile.TemporaryDirectory(prefix="modal_train_job_") as temp_dir:
        job_dir = Path(temp_dir)
        input_dir = job_dir / "input"
        input_dir.mkdir(parents=True, exist_ok=True)
        materialized_training_config = dict(training_config)
        materialized_training_config["artifact_root_dir"] = str(DEPLOY_ARTIFACT_ROOT_DIR)
        materialized_training_config.pop("hpc_training_root_dir", None)
        _write_flat_toml(job_dir / "train_request.toml", materialized_training_config)
        (input_dir / "training_trajectories.sqlite").write_bytes(trajectory_bytes)

        cmd = [
            "torchrun",
            "--nproc_per_node",
            str(num_gpus),
            "--master_port",
            "29500",
            "-m",
            "src_py.train.main_from_config",
            "--job-folder-path",
            str(job_dir),
        ]
        termination_requested = threading.Event()
        child: subprocess.Popen[str] | None = None

        def _on_term(_: int, __: Any) -> None:
            termination_requested.set()
            nonlocal child
            if child is not None and child.poll() is None:
                child.terminate()

        previous_sigterm = signal.signal(signal.SIGTERM, _on_term)
        previous_sigint = signal.signal(signal.SIGINT, _on_term)
        try:
            child = subprocess.Popen(
                cmd,
                cwd="/root/credit_assignment",
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            while child.poll() is None:
                if termination_requested.is_set():
                    break
                time.sleep(0.5)
            stdout_text, stderr_text = child.communicate(timeout=30)
            return_code = child.returncode
        finally:
            signal.signal(signal.SIGTERM, previous_sigterm)
            signal.signal(signal.SIGINT, previous_sigint)

        if termination_requested.is_set():
            return {
                "ok": False,
                "error_code": "CANCELLED_BY_SIGNAL",
                "error": "modal training subprocess received SIGTERM/SIGINT",
            }

        if return_code == 0:
            return {"ok": True, "message": "training completed"}
        return {
            "ok": False,
            "error_code": "TRAIN_PROCESS_FAILED",
            "error": (
                f"training subprocess failed rc={return_code}; "
                f"stdout_tail={stdout_text[-1000:]}; stderr_tail={stderr_text[-1000:]}"
            ),
        }


@app.cls(
    image=training_image,
    gpu=GPU,
    region=REGION,
    startup_timeout=20 * MINUTES,
    min_containers=0,
    max_containers=1,
    timeout=8 * 60 * MINUTES,
    volumes={
        "/mnt/service-state": service_state_volume,
        HF_CACHE_PATH: hf_cache_volume,
    },
)
class ExperimentService:
    @modal.method()
    def train(
        self,
        training_config: dict[str, Any],
        trajectory_bytes: bytes,
        num_gpus: int,
    ) -> dict[str, Any]:
        model_cli_name = str(training_config.get("model_cli_name") or "")
        config_nickname = str(training_config.get("config_nickname") or "")
        epoch = int(training_config.get("epoch", 0))
        if (
            model_cli_name != DEPLOY_MODEL_CLI_NAME
            or config_nickname != DEPLOY_CONFIG_NICKNAME
            or epoch != DEPLOY_EPOCH
        ):
            return {
                "ok": False,
                "error_code": "EXPERIMENT_KEY_MISMATCH",
                "error": (
                    "training config does not match ExperimentService identity: "
                    f"expected=({DEPLOY_MODEL_CLI_NAME},{DEPLOY_CONFIG_NICKNAME},{DEPLOY_EPOCH}) "
                    f"actual=({model_cli_name},{config_nickname},{epoch})"
                ),
            }
        return _run_training_subprocess(training_config, trajectory_bytes, num_gpus)


@app.local_entrypoint()
def show_url() -> None:
    print("Deploy with: modal deploy modal_training_app.py")
