import os
import signal
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import modal

from src_py.load_model_to_path import ensure_model_snapshot
from src_py.train.pathing import model_parent_dir, resolve_artifact_root_dir

MINUTES = 60
APP_NAME = "credit-assignment-training-service"
REGION = "us-west"
GPU = "H100:1"
MODEL_NAME = os.environ.get("SGLANG_MODEL_NAME", "Qwen/Qwen3.5-4B")

HF_CACHE_PATH = "/mnt/hf-cache"
SERVICE_STATE_ROOT = Path("/mnt/service-state")

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


def _run_training_subprocess(
    training_config: dict[str, Any],
    trajectory_bytes: bytes,
    num_gpus: int,
    model_name: str,
) -> dict[str, Any]:
    if num_gpus != 1:
        return {
            "ok": False,
            "error_code": "MODAL_GPU_COUNT_UNSUPPORTED",
            "error": "modal training currently supports num_gpus=1 only",
        }

    epoch = int(training_config.get("epoch", 0))
    if epoch == 0:
        artifact_root_dir = resolve_artifact_root_dir(training_config)
        model_cli_name = str(training_config["model_cli_name"])
        config_nickname = str(training_config["config_nickname"])
        parent_dir = model_parent_dir(artifact_root_dir, model_cli_name, config_nickname, epoch)
        if not (parent_dir / "model").exists():
            ensure_model_snapshot(parent_dir, model_name)

    with tempfile.TemporaryDirectory(prefix="modal_train_job_") as temp_dir:
        job_dir = Path(temp_dir)
        input_dir = job_dir / "input"
        input_dir.mkdir(parents=True, exist_ok=True)
        _write_flat_toml(job_dir / "train_request.toml", training_config)
        (input_dir / "training_trajectories.sqlite").write_bytes(trajectory_bytes)

        cmd = [
            "python",
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
            _stdout_text, stderr_text = child.communicate(timeout=30)
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
            "error": f"training subprocess failed rc={return_code}; stderr={stderr_text[-1000:]}",
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
    model_cli_name: str = modal.parameter()
    config_nickname: str = modal.parameter()
    model_name: str = modal.parameter(default=MODEL_NAME)

    @modal.method()
    def train(
        self,
        training_config: dict[str, Any],
        trajectory_bytes: bytes,
        num_gpus: int,
    ) -> dict[str, Any]:
        model_cli_name = str(training_config.get("model_cli_name") or "")
        config_nickname = str(training_config.get("config_nickname") or "")
        if model_cli_name != self.model_cli_name or config_nickname != self.config_nickname:
            return {
                "ok": False,
                "error_code": "EXPERIMENT_KEY_MISMATCH",
                "error": (
                    "training config does not match ExperimentService identity: "
                    f"expected=({self.model_cli_name},{self.config_nickname}) "
                    f"actual=({model_cli_name},{config_nickname})"
                ),
            }
        return _run_training_subprocess(training_config, trajectory_bytes, num_gpus, self.model_name)


@app.local_entrypoint()
def show_url() -> None:
    print("Deploy with: modal deploy modal_training_app.py")
