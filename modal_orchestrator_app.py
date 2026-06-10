import hashlib
import json
import re
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import modal

MINUTES = 60
APP_NAME = "credit-assignment-orchestrator-service"
REGION = "us-west"

ORCHESTRATOR_CONFIG_RELATIVE_PATH = Path("src_py/modal/orchestrator_config.json")


def _extract_required_cli_arg(cli_args: list[str], flag_name: str) -> str:
    index = 0
    while index < len(cli_args):
        arg = cli_args[index]
        if arg == flag_name:
            if index + 1 >= len(cli_args):
                raise RuntimeError(
                    f"orchestrator config missing value after {flag_name}"
                )
            value = cli_args[index + 1].strip()
            if not value:
                raise RuntimeError(
                    f"orchestrator config value after {flag_name} must be non-empty"
                )
            return value
        prefixed_flag = f"{flag_name}="
        if arg.startswith(prefixed_flag):
            value = arg[len(prefixed_flag) :].strip()
            if not value:
                raise RuntimeError(
                    f"orchestrator config value for {flag_name} must be non-empty"
                )
            return value
        index += 1
    raise RuntimeError(f"orchestrator config must include {flag_name}")


MAX_MODAL_OBJECT_NAME_LENGTH = 64


def _sanitize_modal_name_component(value: str) -> str:
    sanitized = re.sub(r"[^a-z0-9-]+", "-", value.lower())
    sanitized = re.sub(r"-+", "-", sanitized).strip("-")
    if not sanitized:
        raise RuntimeError(f"invalid empty Modal name component derived from {value!r}")
    return sanitized


def _experiment_service_state_volume_name(
    model_cli_name: str, config_nickname: str
) -> str:
    prefix = "credit-assignment-modal-service-state"
    model_component = _sanitize_modal_name_component(model_cli_name)
    config_component = _sanitize_modal_name_component(config_nickname)
    full_name = f"{prefix}-{model_component}-{config_component}"
    if len(full_name) <= MAX_MODAL_OBJECT_NAME_LENGTH:
        return full_name

    digest = hashlib.sha1(
        f"{model_cli_name}\0{config_nickname}".encode("utf-8")
    ).hexdigest()[:10]
    remaining = MAX_MODAL_OBJECT_NAME_LENGTH - len(prefix) - len(digest) - 3
    if remaining <= 2:
        raise RuntimeError(
            "Modal volume name prefix is too long to derive a per-experiment volume name"
        )
    model_budget = max(1, remaining // 2)
    config_budget = max(1, remaining - model_budget)
    truncated_model = model_component[:model_budget].rstrip("-") or model_component[:1]
    truncated_config = (
        config_component[:config_budget].rstrip("-") or config_component[:1]
    )
    return f"{prefix}-{truncated_model}-{truncated_config}-{digest}"


def _load_orchestrator_cli_args() -> list[str]:
    candidate_paths = [
        Path("/workspace") / ORCHESTRATOR_CONFIG_RELATIVE_PATH,
        Path(__file__).resolve().parent / ORCHESTRATOR_CONFIG_RELATIVE_PATH,
    ]
    config_path = next((path for path in candidate_paths if path.is_file()), None)
    if config_path is None:
        searched = ", ".join(str(path) for path in candidate_paths)
        raise RuntimeError(
            "missing orchestrator config JSON; write orchestrator CLI args before deploy/invoke; "
            f"searched: {searched}"
        )
    try:
        raw = config_path.read_text(encoding="utf-8")
    except OSError as error:
        raise RuntimeError(
            f"failed to read orchestrator config JSON at {config_path}: {error}"
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
        raise RuntimeError(
            "orchestrator config must be a JSON array or object with 'args'"
        )

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
    num_gpus_raw = _extract_required_cli_arg(cli_args, "--num-gpus")

    try:
        num_gpus = int(num_gpus_raw)
    except ValueError as error:
        raise RuntimeError(
            f"orchestrator config must include a valid --num-gpus integer: {error}"
        ) from error

    if num_gpus <= 0:
        raise RuntimeError(
            f"orchestrator config --num-gpus must be positive, got {num_gpus}"
        )
    return num_gpus


DEPLOY_ORCHESTRATOR_CLI_ARGS = _load_orchestrator_cli_args()
DEPLOY_MODEL_CLI_NAME = _extract_required_cli_arg(
    DEPLOY_ORCHESTRATOR_CLI_ARGS, "--model-cli-name"
)
DEPLOY_CONFIG_NICKNAME = _extract_required_cli_arg(
    DEPLOY_ORCHESTRATOR_CLI_ARGS, "--config-nickname"
)
DEPLOY_NUM_GPUS = _extract_num_gpus(DEPLOY_ORCHESTRATOR_CLI_ARGS)
GPU = f"H100:{DEPLOY_NUM_GPUS}"
SERVICE_STATE_VOLUME_NAME = _experiment_service_state_volume_name(
    DEPLOY_MODEL_CLI_NAME, DEPLOY_CONFIG_NICKNAME
)
service_state_volume = modal.Volume.from_name(
    SERVICE_STATE_VOLUME_NAME,
    create_if_missing=True,
)


def _print_service_state_volume_status() -> None:
    print(
        "[orchestrate] service state volume "
        f"model_cli_name={DEPLOY_MODEL_CLI_NAME} "
        f"config_nickname={DEPLOY_CONFIG_NICKNAME} "
        f"volume_name={SERVICE_STATE_VOLUME_NAME}"
    )


def _print_workspace_env_file_status() -> None:
    env_path = Path("/workspace/.env")
    if env_path.is_file():
        print(f"[orchestrate] found workspace env file at {env_path}")
    else:
        print(f"[orchestrate] workspace env file missing at {env_path}")


def _print_sglang_env_package_versions() -> None:
    sglang_python = Path("/workspace/pyprojects/sglang/.venv/bin/python")
    if not sglang_python.is_file():
        print(f"[orchestrate] missing sglang python executable at {sglang_python}")
        return

    freeze_cmd = ["uv", "pip", "freeze", "--python", str(sglang_python)]
    result = subprocess.run(
        freeze_cmd,
        cwd="/workspace",
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        install_pip_cmd = [
            "uv",
            "pip",
            "install",
            "--python",
            str(sglang_python),
            "pip",
        ]
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
        stderr = result.stderr.strip()
        print(
            "[orchestrate] failed to list sglang package versions "
            f"(rc={result.returncode}): {stderr}"
        )
        return

    print("[orchestrate] pyprojects/sglang package versions (pip freeze):")
    output = result.stdout.strip()
    if output:
        print(output)
    else:
        print("[orchestrate] (no packages reported)")


def _run_orchestrator_subprocess(cli_args: list[str]) -> dict[str, Any]:
    cmd = ["cargo", "run", "--bin", "bin_orchestrator", "--", *cli_args]

    termination_requested = threading.Event()
    child: subprocess.Popen[bytes] | None = None

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
        )
        assert child is not None
        while child.poll() is None:
            if termination_requested.is_set():
                break
            time.sleep(0.5)
        return_code = child.wait(timeout=30)
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
        signal.signal(signal.SIGINT, previous_sigint)

    if termination_requested.is_set():
        raise RuntimeError(
            "CANCELLED_BY_SIGNAL: modal orchestrator subprocess received SIGTERM/SIGINT"
        )

    if return_code == 0:
        return {"ok": True, "message": "orchestrator completed"}
    raise RuntimeError(
        "ORCHESTRATOR_PROCESS_FAILED: "
        f"orchestrator subprocess failed rc={return_code}; "
        "stdout/stderr are streamed directly to container logs"
    )


orchestrator_image = modal.Image.from_dockerfile(
    "Dockerfile.modal-mirror",
    ignore=modal.FilePatternMatcher.from_file(
        str(Path(__file__).with_name(".dockerignore"))
    ),
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
        _print_service_state_volume_status()
        _print_workspace_env_file_status()
        _print_sglang_env_package_versions()
        cli_args = _load_orchestrator_cli_args()
        requested_num_gpus = _extract_num_gpus(cli_args)
        if requested_num_gpus != DEPLOY_NUM_GPUS:
            raise RuntimeError(
                "MODAL_GPU_COUNT_MISMATCH: "
                f"requested num_gpus={requested_num_gpus} but deployed container has "
                f"DEPLOY_NUM_GPUS={DEPLOY_NUM_GPUS}; redeploy modal_orchestrator_app.py "
                "with matching orchestrator config"
            )
        return _run_orchestrator_subprocess(cli_args)


@app.local_entrypoint()
def show_url() -> None:
    print("Deploy with: modal deploy modal_orchestrator_app.py")
