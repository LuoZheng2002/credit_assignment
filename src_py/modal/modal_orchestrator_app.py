import json
import os
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import modal

MINUTES = 60
REGION = "us-west"

ORCHESTRATOR_CONFIG_RELATIVE_PATH = Path("src_py/modal/orchestrator_config.json")
MODAL_RUNTIME_IGNORE_PATH = ".modalignore"
CARGO_NET_RETRY = "10"
CARGO_HTTP_TIMEOUT = "60"


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


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_orchestrator_config_payload() -> dict[str, Any]:
    candidate_paths = [
        Path("/workspace") / ORCHESTRATOR_CONFIG_RELATIVE_PATH,
        _repo_root() / ORCHESTRATOR_CONFIG_RELATIVE_PATH,
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

    config_payload: dict[str, Any]
    if isinstance(payload, list):
        config_payload = {"args": payload}
    elif isinstance(payload, dict):
        config_payload = payload
    else:
        raise RuntimeError(
            "orchestrator config must be a JSON array or object with 'args'"
        )

    args = config_payload.get("args")
    if not isinstance(args, list):
        raise RuntimeError("orchestrator config field 'args' must be a JSON array")
    if not args:
        raise RuntimeError("orchestrator config CLI args array must be non-empty")

    validated_args: list[str] = []
    for index, value in enumerate(args):
        if not isinstance(value, str):
            raise RuntimeError(
                f"orchestrator config args[{index}] must be string, got {type(value)}"
            )
        stripped = value.strip()
        if not stripped:
            raise RuntimeError(f"orchestrator config args[{index}] must be non-empty")
        validated_args.append(stripped)

    service_state_volume_name = config_payload.get("service_state_volume_name")
    if not isinstance(service_state_volume_name, str):
        raise RuntimeError(
            "orchestrator config field 'service_state_volume_name' must be a string"
        )
    service_state_volume_name = service_state_volume_name.strip()
    if not service_state_volume_name:
        raise RuntimeError(
            "orchestrator config field 'service_state_volume_name' must be non-empty"
        )

    app_name = config_payload.get("app_name")
    if not isinstance(app_name, str):
        raise RuntimeError("orchestrator config field 'app_name' must be a string")
    app_name = app_name.strip()
    if not app_name:
        raise RuntimeError("orchestrator config field 'app_name' must be non-empty")

    return {
        "args": validated_args,
        "service_state_volume_name": service_state_volume_name,
        "app_name": app_name,
    }


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


DEPLOY_ORCHESTRATOR_CONFIG = _load_orchestrator_config_payload()
DEPLOY_ORCHESTRATOR_CLI_ARGS = list(DEPLOY_ORCHESTRATOR_CONFIG["args"])
DEPLOY_MODEL_CLI_NAME = _extract_required_cli_arg(
    DEPLOY_ORCHESTRATOR_CLI_ARGS, "--model-cli-name"
)
DEPLOY_CONFIG_NICKNAME = _extract_required_cli_arg(
    DEPLOY_ORCHESTRATOR_CLI_ARGS, "--config-nickname"
)
DEPLOY_MOUNT_DIR = _extract_required_cli_arg(
    DEPLOY_ORCHESTRATOR_CLI_ARGS, "--mount-dir"
)
DEPLOY_NUM_GPUS = _extract_num_gpus(DEPLOY_ORCHESTRATOR_CLI_ARGS)
GPU = f"H200:{DEPLOY_NUM_GPUS}"
APP_NAME = str(DEPLOY_ORCHESTRATOR_CONFIG["app_name"])
SERVICE_STATE_VOLUME_NAME = str(DEPLOY_ORCHESTRATOR_CONFIG["service_state_volume_name"])
service_state_volume = modal.Volume.from_name(
    SERVICE_STATE_VOLUME_NAME,
    create_if_missing=True,
)


def _print_service_state_volume_status() -> None:
    print(
        "[orchestrate] service state volume "
        f"model_cli_name={DEPLOY_MODEL_CLI_NAME} "
        f"config_nickname={DEPLOY_CONFIG_NICKNAME} "
        f"mount_dir={DEPLOY_MOUNT_DIR} "
        f"volume_name={SERVICE_STATE_VOLUME_NAME}"
    )


def _commit_service_state_volume() -> None:
    print(
        "[orchestrate] committing service state volume "
        f"volume_name={SERVICE_STATE_VOLUME_NAME}"
    )
    service_state_volume.commit()
    print(
        "[orchestrate] committed service state volume "
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
            env={
                **os.environ,
                "CARGO_NET_RETRY": CARGO_NET_RETRY,
                "CARGO_HTTP_TIMEOUT": CARGO_HTTP_TIMEOUT,
            },
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


orchestrator_image = (
    modal.Image.from_dockerfile(
        "Dockerfile.modal-mirror",
        context_dir=str(_repo_root()),
    )
    .env(
        {
            "PYTHONPATH": "/workspace:/workspace/research-utility/src_py",
        }
    )
    .add_local_dir(
        _repo_root(),
        remote_path="/workspace",
        ignore=modal.FilePatternMatcher.from_file(
            str(_repo_root() / MODAL_RUNTIME_IGNORE_PATH)
        ),
    )
)

app = modal.App(name=APP_NAME)


@app.cls(
    image=orchestrator_image,
    gpu=GPU,
    cpu=8.0,
    region=REGION,
    startup_timeout=20 * MINUTES,
    min_containers=0,
    max_containers=1,
    timeout=4 * 60 * MINUTES,
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
        cli_args = list(DEPLOY_ORCHESTRATOR_CLI_ARGS)
        requested_num_gpus = _extract_num_gpus(cli_args)
        if requested_num_gpus != DEPLOY_NUM_GPUS:
            raise RuntimeError(
                "MODAL_GPU_COUNT_MISMATCH: "
                f"requested num_gpus={requested_num_gpus} but deployed container has "
                f"DEPLOY_NUM_GPUS={DEPLOY_NUM_GPUS}; redeploy src_py/modal/modal_orchestrator_app.py "
                "with matching orchestrator config"
            )
        try:
            return _run_orchestrator_subprocess(cli_args)
        finally:
            _commit_service_state_volume()


@app.local_entrypoint()
def show_url() -> None:
    print("Deploy with: modal deploy src_py/modal/modal_orchestrator_app.py")
