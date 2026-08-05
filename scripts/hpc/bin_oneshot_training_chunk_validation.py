#!/usr/bin/env python3
"""Submit a diagnostic training-chunk validation SLURM job."""

from __future__ import annotations

import argparse
import subprocess
import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _read_toml(config_path: Path) -> dict[str, object]:
    with config_path.open("rb") as handle:
        return tomllib.load(handle)


def _resolve_config(path: str) -> Path:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    if not config_path.is_file():
        raise FileNotFoundError(f"config file not found: {config_path}")
    return config_path


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config-path", required=True)
    parser.add_argument("--chunk-index", default=None, type=int)
    parser.add_argument(
        "--all-chunks",
        action="store_true",
        help="Validate every generated training chunk for the experiment.",
    )
    parser.add_argument(
        "--phase",
        required=True,
        choices=["all", "rollout", "judge", "score"],
    )
    parser.add_argument("--time", default="00:30:00")
    parser.add_argument("--dependency", default=None)
    parser.add_argument("--job-name", default=None)
    args = parser.parse_args()
    if args.chunk_index is not None and args.all_chunks:
        raise ValueError("pass either --chunk-index or --all-chunks, not both")
    if args.chunk_index is None and not args.all_chunks:
        args.all_chunks = True

    config_path = _resolve_config(args.config_path)
    config = _read_toml(config_path)
    model_cli_name = _require_str(config, "model_cli_name")
    config_nickname = _require_str(config, "config_nickname_training")
    num_gpus = _require_positive_int(config, "num_gpus")
    phase_uses_gpu = args.phase in {"all", "rollout"}
    job_name = (
        args.job_name
        or f"chunkval_{args.phase}_{model_cli_name}_{config_nickname}"
        + (f"_chunk_{args.chunk_index}" if args.chunk_index is not None else "_all")
    )
    notify_start_msg = f"{job_name} started running."
    notify_end_msg = f"{job_name} finished running."
    cmd = [
        "sbatch",
        "--job-name",
        job_name,
        "--account",
        "bfsl-delta-gpu" if phase_uses_gpu else "bfsl-delta-cpu",
        "--output",
        "slurm/logs/training_chunk_validation_%j.out",
        "--error",
        "slurm/logs/training_chunk_validation_%j.err",
        "--time",
        args.time,
    ]
    if not phase_uses_gpu:
        cmd.extend(["--partition", "cpu", "--cpus-per-task", "16", "--mem", "16G"])
    else:
        cmd.extend(
            [
                "--gres",
                f"gpu:nvidia_a100:{num_gpus}",
                "--cpus-per-task",
                "16",
                "--mem",
                "16G",
            ]
        )
    if args.dependency:
        cmd.extend(["--dependency", args.dependency])
    cmd.extend(
        [
            str(REPO_ROOT / "slurm" / "oneshot_training_chunk_validation.slurm"),
            str(config_path),
            args.phase,
            notify_start_msg,
            notify_end_msg,
        ]
    )
    if args.chunk_index is not None:
        cmd.insert(-3, str(args.chunk_index))

    print("Submitting diagnostic training-chunk validation job:")
    print(f"  Config:     {config_path}")
    print(f"  Job name:   {job_name}")
    print(f"  Phase:      {args.phase}")
    print(f"  Chunk:      {args.chunk_index if args.chunk_index is not None else 'all'}")
    print(f"  Account:    {'bfsl-delta-gpu' if phase_uses_gpu else 'bfsl-delta-cpu'}")
    print(f"  Partition:  {'default' if phase_uses_gpu else 'cpu'}")
    print(f"  GPUs:       {num_gpus if phase_uses_gpu else 'none'}")
    print("  CPUs:       16")
    print("  Mem:        16G")
    print(f"  Time:       {args.time}")
    if args.dependency:
        print(f"  Dependency: {args.dependency}")
    return subprocess.run(cmd, cwd=REPO_ROOT, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
