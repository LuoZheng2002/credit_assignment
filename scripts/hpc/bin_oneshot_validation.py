#!/usr/bin/env python3
"""Submit a SLURM job that runs bin_oneshot_validation with the given config."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _hours_to_slurm_time(total_hours: float) -> str:
    total_seconds = int(total_hours * 1.1 * 3600)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _read_toml(config_path: Path) -> dict[str, object]:
    with config_path.open("rb") as handle:
        return tomllib.load(handle)


def _require_str(config: dict[str, object], key: str) -> str:
    value = config.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} is missing or not a non-empty string")
    return value


def _require_positive_int(config: dict[str, object], key: str) -> int:
    value = config.get(key)
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{key} is missing or not a positive integer")
    return value


def _require_positive_number(config: dict[str, object], key: str) -> float:
    value = config.get(key)
    if not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"{key} is missing or not a positive number")
    return float(value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config-path", required=True)
    parser.add_argument("-j", "--job-name", default=None)
    parser.add_argument("--dependency", default=None)
    parser.add_argument("--phase", default="all", choices=["all", "rollout", "judge", "score"])
    parser.add_argument("--epoch-interval", type=int, default=3)
    args = parser.parse_args()

    if args.epoch_interval <= 0:
        parser.error("--epoch-interval must be positive")

    config_path = Path(args.config_path)
    if not config_path.is_absolute():
        config_path = _REPO_ROOT / config_path
    if not config_path.is_file():
        print(f"Error: config file not found: {config_path}", file=sys.stderr)
        return 1

    config = _read_toml(config_path)
    model_cli_name = _require_str(config, "model_cli_name")
    config_nickname = _require_str(config, "config_nickname_training")
    num_gpus = _require_positive_int(config, "num_gpus")
    total_time_limit_hours = _require_positive_number(config, "total_time_limit_hours")
    slurm_time = _hours_to_slurm_time(total_time_limit_hours)
    job_name = args.job_name or f"validation_{model_cli_name}_{config_nickname}"

    (_REPO_ROOT / "slurm" / "logs").mkdir(parents=True, exist_ok=True)
    slurm_script = _REPO_ROOT / "slurm" / "oneshot_validation.slurm"

    print("Submitting SLURM job:")
    print(f"  Config:         {config_path}")
    print(f"  Job name:       {job_name}")
    print(f"  Account/QOS:    bfsl-delta-gpu")
    print(f"  GPUs:           {num_gpus}")
    print(f"  Time limit:     {slurm_time} (raw: {total_time_limit_hours}h + 10% buffer)")
    print(f"  Phase:          {args.phase}")
    print(f"  Epoch interval: {args.epoch_interval}")
    if args.dependency:
        print(f"  Dependency:     {args.dependency}")
    print(f"  Slurm script:   {slurm_script}")

    notify_start_msg = f"validation_{model_cli_name}_{config_nickname} started running."
    notify_end_msg = f"validation_{model_cli_name}_{config_nickname} finished running."
    cmd = [
        "sbatch",
        "--job-name",
        job_name,
        "--account",
        "bfsl-delta-gpu",
        "--output",
        "slurm/logs/validation_%j.out",
        "--error",
        "slurm/logs/validation_%j.err",
        "--gres",
        f"gpu:nvidia_a100:{num_gpus}",
        "--time",
        slurm_time,
    ]
    if args.dependency:
        cmd.extend(["--dependency", args.dependency])
    cmd.extend(
        [
            str(slurm_script),
            str(config_path),
            notify_start_msg,
            notify_end_msg,
            "--phase",
            args.phase,
            "--epoch-interval",
            str(args.epoch_interval),
        ]
    )
    result = subprocess.run(cmd, cwd=str(_REPO_ROOT), check=False)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
