import json
import os
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import modal
from src_py.modal.modal_experiment_paths import experiment_service_state_volume_name

MINUTES = 60

ONESHOT_GENERATION_CONFIG_RELATIVE_PATH = Path(
    "src_py/modal/oneshot_generation_config.json"
)
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
                    f"oneshot generation config missing value after {flag_name}"
                )
            value = cli_args[index + 1].strip()
            if not value:
                raise RuntimeError(
                    f"oneshot generation config value after {flag_name} must be non-empty"
                )
            return value
        prefixed_flag = f"{flag_name}="
        if arg.startswith(prefixed_flag):
            value = arg[len(prefixed_flag) :].strip()
            if not value:
                raise RuntimeError(
                    f"oneshot generation config value for {flag_name} must be non-empty"
                )
            return value
        index += 1
    raise RuntimeError(f"oneshot generation config must include {flag_name}")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_oneshot_generation_config_payload() -> dict[str, Any]:
    candidate_paths = [
        Path("/workspace") / ONESHOT_GENERATION_CONFIG_RELATIVE_PATH,
        _repo_root() / ONESHOT_GENERATION_CONFIG_RELATIVE_PATH,
    ]
    config_path = next((path for path in candidate_paths if path.is_file()), None)
    if config_path is None:
        searched = ", ".join(str(path) for path in candidate_paths)
        raise RuntimeError(
            "missing oneshot generation config JSON; write oneshot generation CLI args before deploy/invoke; "
            f"searched: {searched}"
        )
    try:
        raw = config_path.read_text(encoding="utf-8")
    except OSError as error:
        raise RuntimeError(
            f"failed to read oneshot generation config JSON at {config_path}: {error}"
        ) from error
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"invalid oneshot generation config JSON: {error}") from error

    config_payload: dict[str, Any]
    if isinstance(payload, list):
        config_payload = {"args": payload}
    elif isinstance(payload, dict):
        config_payload = payload
    else:
        raise RuntimeError(
            "oneshot generation config must be a JSON array or object with 'args'"
        )

    args = config_payload.get("args")
    if not isinstance(args, list):
        raise RuntimeError("oneshot generation config field 'args' must be a JSON array")
    if not args:
        raise RuntimeError("oneshot generation config CLI args array must be non-empty")

    validated_args: list[str] = []
    for index, value in enumerate(args):
        if not isinstance(value, str):
            raise RuntimeError(
                f"oneshot generation config args[{index}] must be string, got {type(value)}"
            )
        stripped = value.strip()
        if not stripped:
            raise RuntimeError(
                f"oneshot generation config args[{index}] must be non-empty"
            )
        validated_args.append(stripped)

    service_state_volume_name_generation = config_payload.get(
        "service_state_volume_name_generation"
    )
    if not isinstance(service_state_volume_name_generation, str):
        raise RuntimeError(
            "oneshot generation config field 'service_state_volume_name_generation' must be a string"
        )
    service_state_volume_name_generation = service_state_volume_name_generation.strip()
    if not service_state_volume_name_generation:
        raise RuntimeError(
            "oneshot generation config field 'service_state_volume_name_generation' must be non-empty"
        )

    app_name = config_payload.get("app_name")
    if not isinstance(app_name, str):
        raise RuntimeError("oneshot generation config field 'app_name' must be a string")
    app_name = app_name.strip()
    if not app_name:
        raise RuntimeError("oneshot generation config field 'app_name' must be non-empty")

    gpu_name = config_payload.get("gpu_name")
    if not isinstance(gpu_name, str):
        raise RuntimeError("oneshot generation config field 'gpu_name' must be a string")
    gpu_name = gpu_name.strip()
    if not gpu_name:
        raise RuntimeError("oneshot generation config field 'gpu_name' must be non-empty")

    num_gpus = config_payload.get("num_gpus")
    if not isinstance(num_gpus, int) or isinstance(num_gpus, bool):
        raise RuntimeError(
            "oneshot generation config field 'num_gpus' must be an integer"
        )
    if num_gpus <= 0:
        raise RuntimeError(
            f"oneshot generation config --num-gpus must be positive, got {num_gpus}"
        )

    return {
        "args": validated_args,
        "service_state_volume_name_generation": service_state_volume_name_generation,
        "app_name": app_name,
        "gpu_name": gpu_name,
        "num_gpus": num_gpus,
    }


DEPLOY_ONESHOT_GENERATION_CONFIG = _load_oneshot_generation_config_payload()
DEPLOY_ONESHOT_GENERATION_CLI_ARGS = list(DEPLOY_ONESHOT_GENERATION_CONFIG["args"])
DEPLOY_MODEL_CLI_NAME = _extract_required_cli_arg(
    DEPLOY_ONESHOT_GENERATION_CLI_ARGS, "--model-cli-name"
)
DEPLOY_CONFIG_NICKNAME_ROLLOUT = _extract_required_cli_arg(
    DEPLOY_ONESHOT_GENERATION_CLI_ARGS, "--config-nickname-rollout"
)
DEPLOY_CONFIG_NICKNAME_GENERATION = _extract_required_cli_arg(
    DEPLOY_ONESHOT_GENERATION_CLI_ARGS, "--config-nickname-generation"
)
DEPLOY_ROLLOUT_MOUNT_DIR = _extract_required_cli_arg(
    DEPLOY_ONESHOT_GENERATION_CLI_ARGS, "--rollout-mount-dir"
)
DEPLOY_GENERATION_MOUNT_DIR = _extract_required_cli_arg(
    DEPLOY_ONESHOT_GENERATION_CLI_ARGS, "--generation-mount-dir"
)
DEPLOY_NUM_GPUS = int(DEPLOY_ONESHOT_GENERATION_CONFIG["num_gpus"])
DEPLOY_GPU_NAME = str(DEPLOY_ONESHOT_GENERATION_CONFIG["gpu_name"])
DEPLOY_MODAL_TIME_LIMIT_HRS = float(
    DEPLOY_ONESHOT_GENERATION_CONFIG.get("modal_time_limit_hrs", 12.0)
)
MODAL_TIMEOUT_SECS = int(DEPLOY_MODAL_TIME_LIMIT_HRS * 60 * MINUTES)
GPU = (
    DEPLOY_GPU_NAME if DEPLOY_NUM_GPUS == 1 else f"{DEPLOY_GPU_NAME}:{DEPLOY_NUM_GPUS}"
)
APP_NAME = str(DEPLOY_ONESHOT_GENERATION_CONFIG["app_name"])
SERVICE_STATE_VOLUME_NAME_GENERATION = str(
    DEPLOY_ONESHOT_GENERATION_CONFIG["service_state_volume_name_generation"]
)
ROLL_OUT_VOLUME_NAME = experiment_service_state_volume_name(
    DEPLOY_MODEL_CLI_NAME, DEPLOY_CONFIG_NICKNAME_ROLLOUT, pipeline="rollout"
)
rollout_volume = modal.Volume.from_name(
    ROLL_OUT_VOLUME_NAME,
    create_if_missing=True,
)
generation_volume = modal.Volume.from_name(
    SERVICE_STATE_VOLUME_NAME_GENERATION,
    create_if_missing=True,
)


def _print_volume_status() -> None:
    print(
        "[oneshot_generation] volumes "
        f"model_cli_name={DEPLOY_MODEL_CLI_NAME} "
        f"config_nickname_rollout={DEPLOY_CONFIG_NICKNAME_ROLLOUT} "
        f"config_nickname_generation={DEPLOY_CONFIG_NICKNAME_GENERATION} "
        f"rollout_mount_dir={DEPLOY_ROLLOUT_MOUNT_DIR} "
        f"generation_mount_dir={DEPLOY_GENERATION_MOUNT_DIR} "
        f"gpu={GPU} "
        f"rollout_volume_name={ROLL_OUT_VOLUME_NAME} "
        f"generation_volume_name={SERVICE_STATE_VOLUME_NAME_GENERATION}"
    )


def _commit_generation_volume() -> None:
    print(
        "[oneshot_generation] committing generation volume "
        f"volume_name={SERVICE_STATE_VOLUME_NAME_GENERATION}"
    )
    generation_volume.commit()
    print(
        "[oneshot_generation] committed generation volume "
        f"volume_name={SERVICE_STATE_VOLUME_NAME_GENERATION}"
    )


def _print_workspace_env_file_status() -> None:
    env_path = Path("/workspace/.env")
    if env_path.is_file():
        print(f"[oneshot_generation] found workspace env file at {env_path}")
    else:
        print(f"[oneshot_generation] workspace env file missing at {env_path}")


def _run_oneshot_generation_subprocess(cli_args: list[str]) -> dict[str, Any]:
    cmd = ["cargo", "run", "--bin", "bin_oneshot_generation", "--", *cli_args]

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
            "CANCELLED_BY_SIGNAL: modal oneshot generation subprocess received SIGTERM/SIGINT"
        )

    if return_code == 0:
        return {"ok": True, "message": "oneshot generation completed"}
    raise RuntimeError(
        "ONESHOT_GENERATION_PROCESS_FAILED: "
        f"oneshot generation subprocess failed rc={return_code}; "
        "stdout/stderr are streamed directly to container logs"
    )


oneshot_generation_image = (
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
    image=oneshot_generation_image,
    gpu=GPU,
    cpu=8.0,
    startup_timeout=20 * MINUTES,
    min_containers=0,
    max_containers=1,
    timeout=MODAL_TIMEOUT_SECS,
    volumes={
        "/rollout_volume": rollout_volume,
        "/generation_volume": generation_volume,
    },
)
class OneshotGenerationService:
    @modal.method()
    def run_generation(self) -> dict[str, Any]:
        _print_volume_status()
        _print_workspace_env_file_status()
        cli_args = list(DEPLOY_ONESHOT_GENERATION_CLI_ARGS)
        try:
            return _run_oneshot_generation_subprocess(cli_args)
        finally:
            _commit_generation_volume()


@app.local_entrypoint()
def show_url() -> None:
    print("Deploy with: modal deploy src_py/modal/modal_oneshot_generation_app.py")
