from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _extract_flag_value(script_text: str, flag: str) -> str:
    pattern = rf"{re.escape(flag)}\s+([^\s\\]+)"
    match = re.search(pattern, script_text)
    if match is None:
        raise RuntimeError(f"Failed to find {flag} in orchestrator script")
    return match.group(1)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Write placeholder training deploy config, deploy modal_training_app.py, "
            "and remove the config file"
        )
    )
    parser.add_argument(
        "--orchestrator-script",
        type=str,
        default="scripts/orchestrator/qwen25_notool_modal.sh",
    )
    parser.add_argument(
        "--model-api-name",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
    )
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--artifact-root-dir", type=str, default="/mnt/service-state")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    repo_root = _repo_root()
    sys.path.insert(0, str(repo_root))

    from src_py.modal.training_deployment_common import deployment_name

    orchestrator_script = repo_root / args.orchestrator_script
    if not orchestrator_script.is_file():
        raise RuntimeError(f"Orchestrator script not found: {orchestrator_script}")
    script_text = orchestrator_script.read_text(encoding="utf-8")

    model_cli_name = _extract_flag_value(script_text, "--model-cli-name")
    config_nickname = _extract_flag_value(script_text, "--config-nickname")
    num_gpus_raw = _extract_flag_value(script_text, "--num-gpus")
    num_gpus = int(num_gpus_raw)
    if num_gpus <= 0:
        raise RuntimeError(f"--num-gpus must be positive in {orchestrator_script}, got {num_gpus}")

    deploy_payload = {
        "DEPLOY_MODEL_CLI_NAME": model_cli_name,
        "DEPLOY_MODEL_API_NAME": args.model_api_name,
        "DEPLOY_CONFIG_NICKNAME": config_nickname,
        "DEPLOY_EPOCH": args.epoch,
        "DEPLOY_NUM_GPUS": num_gpus,
        "DEPLOY_ARTIFACT_ROOT_DIR": args.artifact_root_dir,
    }
    deploy_config_path = repo_root / "src_py/modal/training_deploy_config.json"

    app_name = deployment_name(
        model_cli_name=model_cli_name,
        model_api_name=args.model_api_name,
        config_nickname=config_nickname,
        epoch=args.epoch,
    )

    print(f"Writing placeholder deploy config to {deploy_config_path}", flush=True)
    print(f"Deploying Modal training app as: {app_name}", flush=True)

    deploy_config_path.write_text(
        json.dumps(deploy_payload, ensure_ascii=True),
        encoding="utf-8",
    )

    try:
        result = subprocess.run(
            ["uv", "run", "modal", "deploy", "modal_training_app.py", "--name", app_name],
            cwd=str(repo_root),
            check=False,
        )
        return result.returncode
    finally:
        if deploy_config_path.exists():
            deploy_config_path.unlink()
            print(f"Removed placeholder deploy config: {deploy_config_path}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
