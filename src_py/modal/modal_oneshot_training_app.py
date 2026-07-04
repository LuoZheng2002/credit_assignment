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

ONESHOT_TRAINING_CONFIG_RELATIVE_PATH = Path(
    "src_py/modal/oneshot_training_config.json"
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
                    f"oneshot training config missing value after {flag_name}"
                )
            value = cli_args[index + 1].strip()
            if not value:
                raise RuntimeError(
                    f"oneshot training config value after {flag_name} must be non-empty"
                )
            return value
        prefixed_flag = f"{flag_name}="
        if arg.startswith(prefixed_flag):
            value = arg[len(prefixed_flag) :].strip()
            if not value:
                raise RuntimeError(
                    f"oneshot training config value for {flag_name} must be non-empty"
                )
            return value
        index += 1
    raise RuntimeError(f"oneshot training config must include {flag_name}")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_oneshot_training_config_payload() -> dict[str, Any]:
    candidate_paths = [
        Path("/workspace") / ONESHOT_TRAINING_CONFIG_RELATIVE_PATH,
        _repo_root() / ONESHOT_TRAINING_CONFIG_RELATIVE_PATH,
    ]
    config_path = next((path for path in candidate_paths if path.is_file()), None)
    if config_path is None:
        searched = ", ".join(str(path) for path in candidate_paths)
        raise RuntimeError(
            "missing oneshot training config JSON; write oneshot training CLI args before deploy/invoke; "
            f"searched: {searched}"
        )
    try:
        raw = config_path.read_text(encoding="utf-8")
    except OSError as error:
        raise RuntimeError(
            f"failed to read oneshot training config JSON at {config_path}: {error}"
        ) from error
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"invalid oneshot training config JSON: {error}") from error

    config_payload: dict[str, Any]
    if isinstance(payload, list):
        config_payload = {"args": payload}
    elif isinstance(payload, dict):
        config_payload = payload
    else:
        raise RuntimeError(
            "oneshot training config must be a JSON array or object with 'args'"
        )

    args = config_payload.get("args")
    if not isinstance(args, list):
        raise RuntimeError("oneshot training config field 'args' must be a JSON array")
    if not args:
        raise RuntimeError("oneshot training config CLI args array must be non-empty")

    validated_args: list[str] = []
    for index, value in enumerate(args):
        if not isinstance(value, str):
            raise RuntimeError(
                f"oneshot training config args[{index}] must be string, got {type(value)}"
            )
        stripped = value.strip()
        if not stripped:
            raise RuntimeError(
                f"oneshot training config args[{index}] must be non-empty"
            )
        validated_args.append(stripped)

    service_state_volume_name = config_payload.get("service_state_volume_name")
    if not isinstance(service_state_volume_name, str):
        raise RuntimeError(
            "oneshot training config field 'service_state_volume_name' must be a string"
        )
    service_state_volume_name = service_state_volume_name.strip()
    if not service_state_volume_name:
        raise RuntimeError(
            "oneshot training config field 'service_state_volume_name' must be non-empty"
        )

    app_name = config_payload.get("app_name")
    if not isinstance(app_name, str):
        raise RuntimeError("oneshot training config field 'app_name' must be a string")
    app_name = app_name.strip()
    if not app_name:
        raise RuntimeError("oneshot training config field 'app_name' must be non-empty")

    gpu_name = config_payload.get("gpu_name")
    if not isinstance(gpu_name, str):
        raise RuntimeError("oneshot training config field 'gpu_name' must be a string")
    gpu_name = gpu_name.strip()
    if not gpu_name:
        raise RuntimeError("oneshot training config field 'gpu_name' must be non-empty")

    return {
        "args": validated_args,
        "service_state_volume_name": service_state_volume_name,
        "app_name": app_name,
        "gpu_name": gpu_name,
    }


def _extract_num_gpus(cli_args: list[str]) -> int:
    num_gpus_raw = _extract_required_cli_arg(cli_args, "--num-gpus")

    try:
        num_gpus = int(num_gpus_raw)
    except ValueError as error:
        raise RuntimeError(
            f"oneshot training config must include a valid --num-gpus integer: {error}"
        ) from error

    if num_gpus <= 0:
        raise RuntimeError(
            f"oneshot training config --num-gpus must be positive, got {num_gpus}"
        )
    return num_gpus


DEPLOY_ONESHOT_TRAINING_CONFIG = _load_oneshot_training_config_payload()
DEPLOY_ONESHOT_TRAINING_CLI_ARGS = list(DEPLOY_ONESHOT_TRAINING_CONFIG["args"])
DEPLOY_MODEL_CLI_NAME = _extract_required_cli_arg(
    DEPLOY_ONESHOT_TRAINING_CLI_ARGS, "--model-cli-name"
)
DEPLOY_CONFIG_NICKNAME_ROLLOUT = _extract_required_cli_arg(
    DEPLOY_ONESHOT_TRAINING_CLI_ARGS, "--config-nickname-rollout"
)
DEPLOY_CONFIG_NICKNAME_TRAINING = _extract_required_cli_arg(
    DEPLOY_ONESHOT_TRAINING_CLI_ARGS, "--config-nickname-training"
)
DEPLOY_MOUNT_DIR = _extract_required_cli_arg(
    DEPLOY_ONESHOT_TRAINING_CLI_ARGS, "--mount-dir"
)
DEPLOY_NUM_GPUS = _extract_num_gpus(DEPLOY_ONESHOT_TRAINING_CLI_ARGS)
DEPLOY_GPU_NAME = str(DEPLOY_ONESHOT_TRAINING_CONFIG["gpu_name"])
DEPLOY_MODAL_TIME_LIMIT_HRS = float(
    DEPLOY_ONESHOT_TRAINING_CONFIG.get("modal_time_limit_hrs", 12.0)
)
MODAL_TIMEOUT_SECS = int(DEPLOY_MODAL_TIME_LIMIT_HRS * 60 * MINUTES)
GPU = (
    DEPLOY_GPU_NAME if DEPLOY_NUM_GPUS == 1 else f"{DEPLOY_GPU_NAME}:{DEPLOY_NUM_GPUS}"
)
APP_NAME = str(DEPLOY_ONESHOT_TRAINING_CONFIG["app_name"])
SERVICE_STATE_VOLUME_NAME = str(
    DEPLOY_ONESHOT_TRAINING_CONFIG["service_state_volume_name"]
)
service_state_volume = modal.Volume.from_name(
    SERVICE_STATE_VOLUME_NAME,
    create_if_missing=True,
)
rollout_volume_name = experiment_service_state_volume_name(
    DEPLOY_MODEL_CLI_NAME, DEPLOY_CONFIG_NICKNAME_ROLLOUT, pipeline="oneshot-rollout"
)
rollout_volume = modal.Volume.from_name(
    rollout_volume_name,
    create_if_missing=True,
)


def _print_service_state_volume_status() -> None:
    print(
        "[oneshot_training] service state volume "
        f"model_cli_name={DEPLOY_MODEL_CLI_NAME} "
        f"config_nickname_rollout={DEPLOY_CONFIG_NICKNAME_ROLLOUT} "
        f"config_nickname_training={DEPLOY_CONFIG_NICKNAME_TRAINING} "
        f"mount_dir={DEPLOY_MOUNT_DIR} "
        f"gpu={GPU} "
        f"volume_name={SERVICE_STATE_VOLUME_NAME}"
    )


def _commit_service_state_volume() -> None:
    print(
        "[oneshot_training] committing service state volume "
        f"volume_name={SERVICE_STATE_VOLUME_NAME}"
    )
    service_state_volume.commit()
    print(
        "[oneshot_training] committed service state volume "
        f"volume_name={SERVICE_STATE_VOLUME_NAME}"
    )


def _commit_rollout_volume() -> None:
    print(
        "[oneshot_training] committing rollout volume "
        f"volume_name={rollout_volume_name}"
    )
    rollout_volume.commit()
    print(
        f"[oneshot_training] committed rollout volume volume_name={rollout_volume_name}"
    )


def _print_workspace_env_file_status() -> None:
    env_path = Path("/workspace/.env")
    if env_path.is_file():
        print(f"[oneshot_training] found workspace env file at {env_path}")
    else:
        print(f"[oneshot_training] workspace env file missing at {env_path}")


def _print_sglang_env_package_versions() -> None:
    sglang_python = Path("/workspace/pyprojects/sglang/.venv/bin/python")
    if not sglang_python.is_file():
        print(f"[oneshot_training] missing sglang python executable at {sglang_python}")
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
            "[oneshot_training] failed to list sglang package versions "
            f"(rc={result.returncode}): {stderr}"
        )
        return

    print("[oneshot_training] pyprojects/sglang package versions (pip freeze):")
    output = result.stdout.strip()
    if output:
        print(output)
    else:
        print("[oneshot_training] (no packages reported)")


def _run_oneshot_training_subprocess(cli_args: list[str]) -> dict[str, Any]:
    cmd = ["cargo", "run", "--bin", "bin_oneshot_training", "--", *cli_args]

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
            "CANCELLED_BY_SIGNAL: modal oneshot training subprocess received SIGTERM/SIGINT"
        )

    if return_code == 0:
        return {"ok": True, "message": "oneshot training completed"}
    raise RuntimeError(
        "ONESHOT_TRAINING_PROCESS_FAILED: "
        f"oneshot training subprocess failed rc={return_code}; "
        "stdout/stderr are streamed directly to container logs"
    )


oneshot_training_image = (
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
    image=oneshot_training_image,
    gpu=GPU,
    cpu=8.0,
    startup_timeout=20 * MINUTES,
    min_containers=0,
    max_containers=1,
    timeout=MODAL_TIMEOUT_SECS,
    volumes={
        "/volume": service_state_volume,
        "/rollout_volume": rollout_volume,
    },
)
class OneshotTrainingService:
    @modal.method()
    def run_training(self) -> dict[str, Any]:
        _print_service_state_volume_status()
        _print_workspace_env_file_status()
        _print_sglang_env_package_versions()
        cli_args = list(DEPLOY_ONESHOT_TRAINING_CLI_ARGS)
        requested_num_gpus = _extract_num_gpus(cli_args)
        if requested_num_gpus != DEPLOY_NUM_GPUS:
            raise RuntimeError(
                "MODAL_GPU_COUNT_MISMATCH: "
                f"requested num_gpus={requested_num_gpus} but deployed container has "
                f"DEPLOY_NUM_GPUS={DEPLOY_NUM_GPUS}; redeploy src_py/modal/modal_oneshot_training_app.py "
                "with matching oneshot training config"
            )
        try:
            return _run_oneshot_training_subprocess(cli_args)
        finally:
            _commit_service_state_volume()
            _commit_rollout_volume()


@app.local_entrypoint()
def show_url() -> None:
    print("Deploy with: modal deploy src_py/modal/modal_oneshot_training_app.py")
