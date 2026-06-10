from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import modal

APP_NAME = "credit-assignment-orchestrator-service"
CLASS_NAME = "OrchestratorService"
CONFIG_PATH = Path("src_py/modal/orchestrator_config.json")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _write_orchestrator_config(repo_root: Path, cli_args: list[str]) -> Path:
    if not cli_args:
        raise RuntimeError(
            "No orchestrator arguments provided. Pass the same flags you would pass to bin_orchestrator."
        )
    config_path = repo_root / CONFIG_PATH
    config_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"args": cli_args}
    config_path.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")
    return config_path


def _extract_num_gpus(cli_args: list[str]) -> int:
    index = 0
    while index < len(cli_args):
        arg = cli_args[index]
        if arg == "--num-gpus":
            if index + 1 >= len(cli_args):
                raise RuntimeError("Missing value after --num-gpus")
            raw_value = cli_args[index + 1].strip()
            break
        if arg.startswith("--num-gpus="):
            raw_value = arg.split("=", 1)[1].strip()
            break
        index += 1
    else:
        raise RuntimeError(
            "Missing required --num-gpus argument. Ensure orchestrator script passes --num-gpus."
        )

    try:
        num_gpus = int(raw_value)
    except ValueError as error:
        raise RuntimeError(f"--num-gpus must be an integer, got: {raw_value!r}") from error

    if num_gpus <= 0:
        raise RuntimeError(f"--num-gpus must be positive, got: {num_gpus}")
    return num_gpus


def _normalize_compute_backend(cli_args: list[str]) -> list[str]:
    index = 0
    while index < len(cli_args):
        arg = cli_args[index]
        if arg == "--compute-backend":
            if index + 1 >= len(cli_args):
                raise RuntimeError("Missing value after --compute-backend")
            backend = cli_args[index + 1].strip().lower()
            if backend != "modal":
                raise RuntimeError(
                    f"--compute-backend must be 'modal' for this launcher, got: {backend!r}"
                )
            return cli_args
        if arg.startswith("--compute-backend="):
            backend = arg.split("=", 1)[1].strip().lower()
            if backend != "modal":
                raise RuntimeError(
                    f"--compute-backend must be 'modal' for this launcher, got: {backend!r}"
                )
            return cli_args
        index += 1

    return [*cli_args, "--compute-backend", "modal"]


def _deploy_modal_app(repo_root: Path) -> None:
    command = [
        "uv",
        "run",
        "modal",
        "deploy",
        "modal_orchestrator_app.py",
        "--name",
        APP_NAME,
    ]
    result = subprocess.run(command, cwd=str(repo_root), check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Modal deploy failed with exit code {result.returncode}")


def _remove_config_file(config_path: Path) -> None:
    if config_path.exists():
        config_path.unlink()


def _run_orchestration() -> dict[str, object]:
    cls = modal.Cls.from_name(APP_NAME, CLASS_NAME)
    instance = cls()
    result = instance.orchestrate.remote()
    if isinstance(result, dict):
        return result
    return {
        "ok": False,
        "error_code": "INVALID_REMOTE_RESULT",
        "error": f"unexpected orchestrate result type: {type(result)}",
    }


def main() -> int:
    cli_args = _normalize_compute_backend(sys.argv[1:])
    num_gpus = _extract_num_gpus(cli_args)
    repo_root = _repo_root()
    config_path = _write_orchestrator_config(repo_root, cli_args)
    print(f"Wrote orchestrator config: {config_path}", flush=True)
    print(f"Validated orchestrator num_gpus: {num_gpus}", flush=True)
    print("Validated orchestrator compute_backend: modal", flush=True)
    deployed = False
    try:
        _deploy_modal_app(repo_root)
        deployed = True
        print(f"Deployed Modal app: {APP_NAME}", flush=True)
    finally:
        _remove_config_file(config_path)
        print(f"Removed orchestrator config: {config_path}", flush=True)
    if not deployed:
        return 1
    result = _run_orchestration()
    print(f"Orchestrator result: {json.dumps(result, ensure_ascii=True)}", flush=True)
    if bool(result.get("ok", False)):
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
