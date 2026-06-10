import json
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import modal
from jsonargparse import ArgumentParser

MINUTES = 60
APP_NAME = "credit-assignment-orchestrator-service"
REGION = "us-west"

ORCHESTRATOR_CONFIG_PATH = Path("/workspace/src_py/modal/orchestrator_config.json")

service_state_volume = modal.Volume.from_name(
    "credit-assignment-modal-service-state", create_if_missing=True
)


def _load_orchestrator_cli_args() -> list[str]:
    try:
        raw = ORCHESTRATOR_CONFIG_PATH.read_text(encoding="utf-8")
    except FileNotFoundError as error:
        raise RuntimeError(
            "missing /workspace/src_py/modal/orchestrator_config.json; "
            "write orchestrator CLI args before deploy/invoke"
        ) from error
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"invalid orchestrator config JSON: {error}") from error

    args: Any
    if isinstance(payload, list):
        args = payload
    elif isinstance(payload, dict):
        args = payload.get("args")
    else:
        raise RuntimeError("orchestrator config must be a JSON array or object with 'args'")

    if not isinstance(args, list):
        raise RuntimeError("orchestrator config field 'args' must be a JSON array")
    if not args:
        raise RuntimeError("orchestrator config CLI args array must be non-empty")

    result: list[str] = []
    for index, value in enumerate(args):
        if not isinstance(value, str):
            raise RuntimeError(
                f"orchestrator config args[{index}] must be string, got {type(value)}"
            )
        stripped = value.strip()
        if not stripped:
            raise RuntimeError(f"orchestrator config args[{index}] must be non-empty")
        result.append(stripped)
    return result


def _extract_num_gpus(cli_args: list[str]) -> int:
    try:
        parser = ArgumentParser(exit_on_error=False)
        parser.add_argument("--num-gpus", type=int, required=True)
        parsed, _unknown = parser.parse_known_args(cli_args)
        num_gpus = int(parsed.num_gpus)
    except Exception as error:
        raise RuntimeError(
            f"orchestrator config must include a valid --num-gpus integer: {error}"
        ) from error

    if num_gpus <= 0:
        raise RuntimeError(
            f"orchestrator config --num-gpus must be positive, got {num_gpus}"
        )
    return num_gpus


DEPLOY_ORCHESTRATOR_CLI_ARGS = _load_orchestrator_cli_args()
DEPLOY_NUM_GPUS = _extract_num_gpus(DEPLOY_ORCHESTRATOR_CLI_ARGS)
GPU = f"A100:{DEPLOY_NUM_GPUS}"


def _run_orchestrator_subprocess(cli_args: list[str]) -> dict[str, Any]:
    cmd = ["cargo", "run", "--bin", "bin_orchestrator", "--", *cli_args]

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
            cwd="/workspace",
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
            "error": "modal orchestrator subprocess received SIGTERM/SIGINT",
        }

    if return_code == 0:
        return {"ok": True, "message": "orchestrator completed"}
    return {
        "ok": False,
        "error_code": "ORCHESTRATOR_PROCESS_FAILED",
        "error": (
            f"orchestrator subprocess failed rc={return_code}; "
            f"stdout_tail={stdout_text[-1000:]}; stderr_tail={stderr_text[-1000:]}"
        ),
    }


orchestrator_image = modal.Image.from_dockerfile(
    "Dockerfile.modal-mirror",
    ignore=modal.FilePatternMatcher.from_file(str(Path(__file__).with_name(".dockerignore"))),
).env(
    {
        "PYTHONPATH": "/workspace",
    }
)

app = modal.App(name=APP_NAME)


@app.cls(
    image=orchestrator_image,
    gpu=GPU,
    region=REGION,
    startup_timeout=20 * MINUTES,
    min_containers=0,
    max_containers=1,
    timeout=24 * 60 * MINUTES,
    volumes={
        "/volume": service_state_volume,
    },
)
class OrchestratorService:
    @modal.method()
    def orchestrate(self) -> dict[str, Any]:
        cli_args = _load_orchestrator_cli_args()
        requested_num_gpus = _extract_num_gpus(cli_args)
        if requested_num_gpus != DEPLOY_NUM_GPUS:
            return {
                "ok": False,
                "error_code": "MODAL_GPU_COUNT_MISMATCH",
                "error": (
                    f"requested num_gpus={requested_num_gpus} but deployed container has "
                    f"DEPLOY_NUM_GPUS={DEPLOY_NUM_GPUS}; redeploy modal_orchestrator_app.py "
                    "with matching orchestrator config"
                ),
            }
        return _run_orchestrator_subprocess(cli_args)


@app.local_entrypoint()
def show_url() -> None:
    print("Deploy with: modal deploy modal_orchestrator_app.py")
