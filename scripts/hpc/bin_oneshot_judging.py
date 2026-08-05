#!/usr/bin/env python3
"""Submit a CPU SLURM job that runs bin_oneshot_judging."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-tree-msgpack", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--output-tree-judgment-jsonl", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--escalation-jsonl", required=True)
    parser.add_argument("--model-cli-name", required=True)
    parser.add_argument("--dataset-split", required=True, choices=["training", "validation", "testing", "Training", "Validation", "Testing"])
    parser.add_argument("--time", default="00:30:00")
    parser.add_argument("--job-name", default="oneshot_judging")
    parser.add_argument("--account", default="bfsl-delta-cpu")
    parser.add_argument("--partition", default="cpu")
    parser.add_argument("--cpus-per-task", default="16")
    parser.add_argument("--mem", default="16G")
    parser.add_argument("--dependency", default=None)
    args = parser.parse_args()

    log_dir = REPO_ROOT / "slurm" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    notify_start_msg = f"{args.job_name} started running."
    notify_end_msg = f"{args.job_name} finished running."
    cmd = [
        "sbatch",
        "--job-name", args.job_name,
        "--account", args.account,
        "--partition", args.partition,
        "--cpus-per-task", args.cpus_per_task,
        "--mem", args.mem,
        "--time", args.time,
        "--output", "slurm/logs/judging_%j.out",
        "--error", "slurm/logs/judging_%j.err",
        str(REPO_ROOT / "slurm" / "oneshot_judging_cpu.slurm"),
        "--input-tree-msgpack", args.input_tree_msgpack,
        "--output-jsonl", args.output_jsonl,
        "--output-tree-judgment-jsonl", args.output_tree_judgment_jsonl,
        "--cache-dir", args.cache_dir,
        "--escalation-jsonl", args.escalation_jsonl,
        "--model-cli-name", args.model_cli_name,
        "--dataset-split", args.dataset_split,
        "--notify-start-message", notify_start_msg,
        "--notify-end-message", notify_end_msg,
    ]
    if args.dependency:
        time_index = cmd.index("--time")
        cmd[time_index:time_index] = ["--dependency", args.dependency]
    print("Submitting SLURM judging job:")
    print(f"  Job name: {args.job_name}")
    print(f"  Account:  {args.account}")
    print(f"  Partition:{args.partition}")
    print(f"  Time:     {args.time}")
    print(f"  CPUs:     {args.cpus_per_task}")
    print(f"  Mem:      {args.mem}")
    if args.dependency:
        print(f"  Dependency: {args.dependency}")
    return subprocess.run(cmd, cwd=REPO_ROOT, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
