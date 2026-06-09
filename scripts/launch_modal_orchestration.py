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


def _spawn_orchestration() -> str:
    cls = modal.Cls.from_name(APP_NAME, CLASS_NAME)
    instance = cls()
    function_call = instance.orchestrate.spawn()
    object_id = getattr(function_call, "object_id", None)
    if isinstance(object_id, str) and object_id:
        return object_id
    return "unknown"


def main() -> int:
    cli_args = sys.argv[1:]
    repo_root = _repo_root()
    config_path = _write_orchestrator_config(repo_root, cli_args)
    print(f"Wrote orchestrator config: {config_path}", flush=True)
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
    call_id = _spawn_orchestration()
    print(f"Spawned orchestrator call id: {call_id}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
