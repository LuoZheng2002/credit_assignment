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

ONESHOT_ROLLOUT_CONFIG_RELATIVE_PATH = Path("src_py/modal/oneshot_rollout_config.json")
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
                    f"oneshot rollout config missing value after {flag_name}"
                )
            value = cli_args[index + 1].strip()
            if not value:
                raise RuntimeError(
                    f"oneshot rollout config value after {flag_name} must be non-empty"
                )
            return value
        prefixed_flag = f"{flag_name}="
        if arg.startswith(prefixed_flag):
            value = arg[len(prefixed_flag) :].strip()
            if not value:
                raise RuntimeError(
                    f"oneshot rollout config value for {flag_name} must be non-empty"
                )
            return value
        index += 1
    raise RuntimeError(f"oneshot rollout config must include {flag_name}")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_oneshot_rollout_config_payload() -> dict[str, Any]:
    candidate_paths = [
        Path("/workspace") / ONESHOT_ROLLOUT_CONFIG_RELATIVE_PATH,
        _repo_root() / ONESHOT_ROLLOUT_CONFIG_RELATIVE_PATH,
    ]
    config_path = next((path for path in candidate_paths if path.is_file()), None)
    if config_path is None:
        searched = ", ".join(str(path) for path in candidate_paths)
        raise RuntimeError(
            "missing oneshot rollout config JSON; write oneshot rollout CLI args before deploy/invoke; "
            f"searched: {searched}"
        )
    try:
        raw = config_path.read_text(encoding="utf-8")
    except OSError as error:
        raise RuntimeError(
            f"failed to read oneshot rollout config JSON at {config_path}: {error}"
        ) from error
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"invalid oneshot rollout config JSON: {error}") from error

    config_payload: dict[str, Any]
    if isinstance(payload, list):
        config_payload = {"args": payload}
    elif isinstance(payload, dict):
        config_payload = payload
    else:
        raise RuntimeError(
            "oneshot rollout config must be a JSON array or object with 'args'"
        )

    args = config_payload.get("args")
    if not isinstance(args, list):
        raise RuntimeError("oneshot rollout config field 'args' must be a JSON array")
    if not args:
        raise RuntimeError("oneshot rollout config CLI args array must be non-empty")

    validated_args: list[str] = []
    for index, value in enumerate(args):
        if not isinstance(value, str):
            raise RuntimeError(
                f"oneshot rollout config args[{index}] must be string, got {type(value)}"
            )
        stripped = value.strip()
        if not stripped:
            raise RuntimeError(
                f"oneshot rollout config args[{index}] must be non-empty"
            )
        validated_args.append(stripped)

    service_state_volume_name = config_payload.get("service_state_volume_name")
    if not isinstance(service_state_volume_name, str):
        raise RuntimeError(
            "oneshot rollout config field 'service_state_volume_name' must be a string"
        )
    service_state_volume_name = service_state_volume_name.strip()
    if not service_state_volume_name:
        raise RuntimeError(
            "oneshot rollout config field 'service_state_volume_name' must be non-empty"
        )

    app_name = config_payload.get("app_name")
    if not isinstance(app_name, str):
        raise RuntimeError("oneshot rollout config field 'app_name' must be a string")
    app_name = app_name.strip()
    if not app_name:
        raise RuntimeError("oneshot rollout config field 'app_name' must be non-empty")

    gpu_name = config_payload.get("gpu_name")
    if not isinstance(gpu_name, str):
        raise RuntimeError("oneshot rollout config field 'gpu_name' must be a string")
    gpu_name = gpu_name.strip()
    if not gpu_name:
        raise RuntimeError("oneshot rollout config field 'gpu_name' must be non-empty")

    num_gpus = config_payload.get("num_gpus")
    if not isinstance(num_gpus, int) or isinstance(num_gpus, bool):
        raise RuntimeError("oneshot rollout config field 'num_gpus' must be an integer")
    if num_gpus <= 0:
        raise RuntimeError(
            f"oneshot rollout config --num-gpus must be positive, got {num_gpus}"
        )

    return {
        "args": validated_args,
        "service_state_volume_name": service_state_volume_name,
        "app_name": app_name,
        "gpu_name": gpu_name,
        "num_gpus": num_gpus,
    }


DEPLOY_ONESHOT_ROLLOUT_CONFIG = _load_oneshot_rollout_config_payload()
DEPLOY_ONESHOT_ROLLOUT_CLI_ARGS = list(DEPLOY_ONESHOT_ROLLOUT_CONFIG["args"])
DEPLOY_MODEL_CLI_NAME = _extract_required_cli_arg(
    DEPLOY_ONESHOT_ROLLOUT_CLI_ARGS, "--model-cli-name"
)
DEPLOY_CONFIG_NICKNAME_ROLLOUT = _extract_required_cli_arg(
    DEPLOY_ONESHOT_ROLLOUT_CLI_ARGS, "--config-nickname-rollout"
)
DEPLOY_MOUNT_DIR = _extract_required_cli_arg(
    DEPLOY_ONESHOT_ROLLOUT_CLI_ARGS, "--mount-dir"
)
DEPLOY_NUM_GPUS = int(DEPLOY_ONESHOT_ROLLOUT_CONFIG["num_gpus"])
DEPLOY_GPU_NAME = str(DEPLOY_ONESHOT_ROLLOUT_CONFIG["gpu_name"])
DEPLOY_MODAL_TIME_LIMIT_HRS = float(
    DEPLOY_ONESHOT_ROLLOUT_CONFIG.get("modal_time_limit_hrs", 12.0)
)
MODAL_TIMEOUT_SECS = int(DEPLOY_MODAL_TIME_LIMIT_HRS * 60 * MINUTES)
GPU = (
    DEPLOY_GPU_NAME if DEPLOY_NUM_GPUS == 1 else f"{DEPLOY_GPU_NAME}:{DEPLOY_NUM_GPUS}"
)
APP_NAME = str(DEPLOY_ONESHOT_ROLLOUT_CONFIG["app_name"])
SERVICE_STATE_VOLUME_NAME = str(
    DEPLOY_ONESHOT_ROLLOUT_CONFIG["service_state_volume_name"]
)
service_state_volume = modal.Volume.from_name(
    SERVICE_STATE_VOLUME_NAME,
    create_if_missing=True,
)


def _print_service_state_volume_status() -> None:
    print(
        "[oneshot_rollout] service state volume "
        f"model_cli_name={DEPLOY_MODEL_CLI_NAME} "
        f"config_nickname_rollout={DEPLOY_CONFIG_NICKNAME_ROLLOUT} "
        f"mount_dir={DEPLOY_MOUNT_DIR} "
        f"gpu={GPU} "
        f"volume_name={SERVICE_STATE_VOLUME_NAME}"
    )


def _commit_service_state_volume() -> None:
    print(
        "[oneshot_rollout] committing service state volume "
        f"volume_name={SERVICE_STATE_VOLUME_NAME}"
    )
    service_state_volume.commit()
    print(
        "[oneshot_rollout] committed service state volume "
        f"volume_name={SERVICE_STATE_VOLUME_NAME}"
    )


def _print_workspace_env_file_status() -> None:
    env_path = Path("/workspace/.env")
    if env_path.is_file():
        print(f"[oneshot_rollout] found workspace env file at {env_path}")
    else:
        print(f"[oneshot_rollout] workspace env file missing at {env_path}")


def _run_oneshot_rollout_subprocess(cli_args: list[str]) -> dict[str, Any]:
    cmd = ["cargo", "run", "--bin", "bin_oneshot_rollout", "--", *cli_args]

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
            "CANCELLED_BY_SIGNAL: modal oneshot rollout subprocess received SIGTERM/SIGINT"
        )

    if return_code == 0:
        return {"ok": True, "message": "oneshot rollout completed"}
    raise RuntimeError(
        "ONESHOT_ROLLOUT_PROCESS_FAILED: "
        f"oneshot rollout subprocess failed rc={return_code}; "
        "stdout/stderr are streamed directly to container logs"
    )


oneshot_rollout_image = (
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
    image=oneshot_rollout_image,
    gpu=GPU,
    cpu=8.0,
    startup_timeout=20 * MINUTES,
    min_containers=0,
    max_containers=1,
    timeout=MODAL_TIMEOUT_SECS,
    volumes={
        "/volume": service_state_volume,
    },
)
class OneshotRolloutService:
    @modal.method()
    def run_rollout(self) -> dict[str, Any]:
        _print_service_state_volume_status()
        _print_workspace_env_file_status()
        cli_args = list(DEPLOY_ONESHOT_ROLLOUT_CLI_ARGS)
        try:
            return _run_oneshot_rollout_subprocess(cli_args)
        finally:
            _commit_service_state_volume()


@app.local_entrypoint()
def show_url() -> None:
    print("Deploy with: modal deploy src_py/modal/modal_oneshot_rollout_app.py")
