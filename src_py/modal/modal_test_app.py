import json
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import modal
from src_py.modal.modal_experiment_paths import experiment_service_state_volume_name

MINUTES = 60
REGION = "us-west"

TEST_CONFIG_RELATIVE_PATH = Path("src_py/modal/test_config.json")
TEST_CONFIG_ABSOLUTE_PATH = Path("/workspace") / TEST_CONFIG_RELATIVE_PATH
MODAL_RUNTIME_IGNORE_PATH = ".modalignore"


def _test_app_name(model_cli_name: str, config_nickname: str) -> str:
    return f"credit-assignment-test-{model_cli_name}-{config_nickname}"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _require_dict_value(payload: dict[str, Any], key: str, *, context: str) -> Any:
    if key not in payload:
        raise RuntimeError(f"{context} missing required field '{key}'")
    return payload[key]


def _require_string(payload: dict[str, Any], key: str, *, context: str) -> str:
    value = _require_dict_value(payload, key, context=context)
    if not isinstance(value, str):
        raise RuntimeError(f"{context} field '{key}' must be a string")
    value = value.strip()
    if not value:
        raise RuntimeError(f"{context} field '{key}' must be non-empty")
    return value


def _require_int(payload: dict[str, Any], key: str, *, context: str) -> int:
    value = _require_dict_value(payload, key, context=context)
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(f"{context} field '{key}' must be an integer")
    return int(value)


def _require_bool(payload: dict[str, Any], key: str, *, context: str) -> bool:
    value = _require_dict_value(payload, key, context=context)
    if not isinstance(value, bool):
        raise RuntimeError(f"{context} field '{key}' must be a boolean")
    return value


def _load_test_configs_payload() -> list[dict[str, Any]]:
    candidate_paths = [
        Path("/workspace") / TEST_CONFIG_RELATIVE_PATH,
        _repo_root() / TEST_CONFIG_RELATIVE_PATH,
    ]
    config_path = next((path for path in candidate_paths if path.is_file()), None)
    if config_path is None:
        searched = ", ".join(str(path) for path in candidate_paths)
        raise RuntimeError(
            "missing test config JSON; write test configs before deploy/invoke; "
            f"searched: {searched}"
        )
    try:
        raw = config_path.read_text(encoding="utf-8")
    except OSError as error:
        raise RuntimeError(
            f"failed to read test config JSON at {config_path}: {error}"
        ) from error
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"invalid test config JSON: {error}") from error

    if not isinstance(payload, list):
        raise RuntimeError("test config JSON must be a list of testing configs")
    if not payload:
        raise RuntimeError("test config JSON list must be non-empty")

    normalized: list[dict[str, Any]] = []
    for index, entry in enumerate(payload):
        if not isinstance(entry, dict):
            raise RuntimeError(
                f"test config entry[{index}] must be an object, got {type(entry)}"
            )
        context = f"test config entry[{index}]"
        normalized.append(
            {
                "model_cli_name": _require_string(
                    entry, "model_cli_name", context=context
                ),
                "config_nickname": _require_string(
                    entry, "config_nickname", context=context
                ),
                "testing_rollout_config_path": _require_string(
                    entry, "testing_rollout_config_path", context=context
                ),
                "epoch": _require_int(entry, "epoch", context=context),
                "total_epochs": _require_int(entry, "total_epochs", context=context),
                "max_rollout_concurrency": _require_int(
                    entry, "max_rollout_concurrency", context=context
                ),
                "ui": _require_bool(entry, "ui", context=context),
                "rollout_time_limit_secs": _require_int(
                    entry, "rollout_time_limit_secs", context=context
                ),
                "max_python_processes": _require_int(
                    entry, "max_python_processes", context=context
                ),
                "num_gpus": _require_int(entry, "num_gpus", context=context),
                "mount_dir": _require_string(entry, "mount_dir", context=context),
            }
        )

    reference = normalized[0]
    for index, entry in enumerate(normalized[1:], start=1):
        for key in (
            "max_rollout_concurrency",
            "ui",
            "rollout_time_limit_secs",
            "max_python_processes",
            "num_gpus",
        ):
            if entry[key] != reference[key]:
                raise RuntimeError(
                    f"test config entry[{index}] field '{key}' must match entry[0]; "
                    f"got {entry[key]!r} vs {reference[key]!r}"
                )
    return normalized


def _build_service_state_volume_mounts(
    test_configs: list[dict[str, Any]],
) -> list[tuple[str, str, modal.Volume]]:
    mounts: list[tuple[str, str, modal.Volume]] = []
    seen_mount_dirs: set[str] = set()
    for entry in test_configs:
        mount_dir = entry["mount_dir"]
        if mount_dir in seen_mount_dirs:
            continue
        volume_name = experiment_service_state_volume_name(
            entry["model_cli_name"], entry["config_nickname"]
        )
        mounts.append(
            (
                mount_dir,
                volume_name,
                modal.Volume.from_name(volume_name, create_if_missing=True),
            )
        )
        seen_mount_dirs.add(mount_dir)
    return mounts


DEPLOY_TEST_CONFIGS = _load_test_configs_payload()
DEPLOY_SERVICE_STATE_VOLUME_MOUNTS = _build_service_state_volume_mounts(
    DEPLOY_TEST_CONFIGS
)
DEPLOY_TEST_CONFIG = DEPLOY_TEST_CONFIGS[0]
DEPLOY_MODEL_CLI_NAME = DEPLOY_TEST_CONFIG["model_cli_name"]
DEPLOY_CONFIG_NICKNAME = DEPLOY_TEST_CONFIG["config_nickname"]
DEPLOY_NUM_GPUS = _require_int(
    DEPLOY_TEST_CONFIG, "num_gpus", context="deploy testing config"
)
GPU = f"L40S:{DEPLOY_NUM_GPUS}"
APP_NAME = _test_app_name(DEPLOY_MODEL_CLI_NAME, DEPLOY_CONFIG_NICKNAME)


def _print_service_state_volume_status() -> None:
    for mount_dir, volume_name, _volume in DEPLOY_SERVICE_STATE_VOLUME_MOUNTS:
        print(
            "[test] service state volume "
            f"mount_dir={mount_dir} "
            f"volume_name={volume_name}"
        )


def _commit_service_state_volume() -> None:
    for mount_dir, volume_name, volume in DEPLOY_SERVICE_STATE_VOLUME_MOUNTS:
        print(
            "[test] committing service state volume "
            f"mount_dir={mount_dir} volume_name={volume_name}"
        )
        volume.commit()
        print(
            "[test] committed service state volume "
            f"mount_dir={mount_dir} volume_name={volume_name}"
        )


def _print_workspace_env_file_status() -> None:
    env_path = Path("/workspace/.env")
    if env_path.is_file():
        print(f"[test] found workspace env file at {env_path}")
    else:
        print(f"[test] workspace env file missing at {env_path}")


def _print_sglang_env_package_versions() -> None:
    sglang_python = Path("/workspace/pyprojects/sglang/.venv/bin/python")
    if not sglang_python.is_file():
        print(f"[test] missing sglang python executable at {sglang_python}")
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
            "[test] failed to list sglang package versions "
            f"(rc={result.returncode}): {stderr}"
        )
        return

    print("[test] pyprojects/sglang package versions (pip freeze):")
    output = result.stdout.strip()
    if output:
        print(output)
    else:
        print("[test] (no packages reported)")


def _build_runtime_cli_args() -> list[str]:
    runtime_args = [
        "--testing-configs-path",
        str(TEST_CONFIG_ABSOLUTE_PATH),
        "--max-rollout-concurrency",
        str(DEPLOY_TEST_CONFIG["max_rollout_concurrency"]),
        "--rollout-time-limit-secs",
        str(DEPLOY_TEST_CONFIG["rollout_time_limit_secs"]),
        "--max-python-processes",
        str(DEPLOY_TEST_CONFIG["max_python_processes"]),
        "--num-gpus",
        str(DEPLOY_NUM_GPUS),
    ]
    if DEPLOY_TEST_CONFIG["ui"]:
        runtime_args.append("--ui")
    return runtime_args


def _run_test_subprocess(runtime_cli_args: list[str]) -> dict[str, Any]:
    cmd = ["cargo", "run", "--bin", "bin_run_test", "--", *runtime_cli_args]

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
            "CANCELLED_BY_SIGNAL: modal test subprocess received SIGTERM/SIGINT"
        )

    if return_code == 0:
        return {"ok": True, "message": "test completed"}
    raise RuntimeError(
        "TEST_PROCESS_FAILED: "
        f"test subprocess failed rc={return_code}; "
        "stdout/stderr are streamed directly to container logs"
    )


test_image = (
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
    image=test_image,
    gpu=GPU,
    region=REGION,
    startup_timeout=20 * MINUTES,
    min_containers=0,
    max_containers=1,
    timeout=4 * 60 * MINUTES,
    volumes={
        mount_dir: volume
        for mount_dir, _volume_name, volume in DEPLOY_SERVICE_STATE_VOLUME_MOUNTS
    },
)
class TestService:
    @modal.method()
    def run_test(self) -> dict[str, Any]:
        _print_service_state_volume_status()
        _print_workspace_env_file_status()
        _print_sglang_env_package_versions()
        runtime_cli_args = _build_runtime_cli_args()
        try:
            return _run_test_subprocess(runtime_cli_args)
        finally:
            _commit_service_state_volume()


@app.local_entrypoint()
def show_url() -> None:
    print("Deploy with: modal deploy src_py/modal/modal_test_app.py")
