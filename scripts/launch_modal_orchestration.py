from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import modal

APP_NAME = "credit-assignment-orchestrator-service"
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
        raise RuntimeError(
            f"--num-gpus must be an integer, got: {raw_value!r}"
        ) from error

    if num_gpus <= 0:
        raise RuntimeError(f"--num-gpus must be positive, got: {num_gpus}")
    return num_gpus


def _strip_legacy_compute_backend(cli_args: list[str]) -> list[str]:
    normalized: list[str] = []
    index = 0
    removed = False
    while index < len(cli_args):
        arg = cli_args[index]
        if arg == "--compute-backend":
            if index + 1 >= len(cli_args):
                raise RuntimeError("Missing value after --compute-backend")
            removed = True
            index += 2
            continue
        if arg.startswith("--compute-backend="):
            removed = True
            index += 1
            continue
        normalized.append(arg)
        index += 1
    if removed:
        print(
            "Ignoring legacy --compute-backend flag; orchestration now always uses local wrapper-managed runtime paths.",
            flush=True,
        )
    return normalized


def _remove_config_file(config_path: Path) -> None:
    if config_path.exists():
        config_path.unlink()


def _run_orchestration(repo_root: Path) -> dict[str, object]:
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

    orchestrator_module = importlib.import_module("modal_orchestrator_app")
    app = getattr(orchestrator_module, "app")
    service_cls = getattr(orchestrator_module, "OrchestratorService")

    with modal.enable_output():
        with app.run():
            instance = service_cls()
            result = instance.orchestrate.remote()

    if isinstance(result, dict):
        return result
    return {
        "ok": False,
        "error_code": "INVALID_REMOTE_RESULT",
        "error": f"unexpected orchestrate result type: {type(result)}",
    }


def main() -> int:
    cli_args = _strip_legacy_compute_backend(sys.argv[1:])
    num_gpus = _extract_num_gpus(cli_args)
    repo_root = _repo_root()
    config_path = _write_orchestrator_config(repo_root, cli_args)
    print(f"Wrote orchestrator config: {config_path}", flush=True)
    print(f"Validated orchestrator num_gpus: {num_gpus}", flush=True)
    print(
        "Validated orchestrator runtime: local wrapper-managed inference/training",
        flush=True,
    )
    try:
        print(
            f"Running Modal orchestration app in a single ephemeral step: {APP_NAME}",
            flush=True,
        )
        result = _run_orchestration(repo_root)
    finally:
        _remove_config_file(config_path)
        print(f"Removed orchestrator config: {config_path}", flush=True)
    print(f"Orchestrator result: {json.dumps(result, ensure_ascii=True)}", flush=True)
    if bool(result.get("ok", False)):
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
