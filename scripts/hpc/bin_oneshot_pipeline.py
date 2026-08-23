#!/usr/bin/env python3
"""Submit the chunked one-shot pipeline with explicit SLURM dependencies."""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

REPO_ROOT = Path(__file__).resolve().parents[2]
CPU_CPUS_PER_TASK = "16"
CPU_MEM = "16G"
CPU_TIME_LIMIT = "00:30:00"


def _read_toml(path: Path) -> dict[str, object]:
    with path.open("rb") as handle:
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


def _kl_beta(config: dict[str, object]) -> float:
    hyperparameters = config.get("training_hyperparameters")
    if not isinstance(hyperparameters, dict):
        return 0.0
    value = hyperparameters.get("kl_beta", 0.0)
    if not isinstance(value, (int, float)) or value < 0:
        raise ValueError("training_hyperparameters.kl_beta must be a non-negative number")
    return float(value)


def _hours_to_slurm_time(total_hours: float) -> str:
    total_seconds = int(total_hours * 1.1 * 3600)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _resolve_config(path: str) -> Path:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    if not config_path.is_file():
        raise FileNotFoundError(f"config file not found: {config_path}")
    return config_path


def _submit(cmd: list[str]) -> str:
    print("Submitting:")
    print("  " + " ".join(cmd))
    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.stdout:
        print(result.stdout.rstrip())
    if result.stderr:
        print(result.stderr.rstrip(), file=sys.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"sbatch failed with code {result.returncode}")
    job_id = result.stdout.strip().split(";")[0].strip()
    if not job_id:
        raise RuntimeError(f"failed to parse sbatch job id from stdout: {result.stdout!r}")
    return job_id


def _run_login_smoke(cmd: list[str]) -> None:
    print("Login smoke:")
    print("  " + " ".join(cmd))
    result = subprocess.run(cmd, cwd=REPO_ROOT, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"login smoke failed with code {result.returncode}")


def _cargo_smoke(binary: str, *args: str) -> list[str]:
    return [
        str(REPO_ROOT / "scripts" / "hpc" / "cargo_run_with_rebuild.sh"),
        binary,
        *args,
        "--login-smoke",
    ]


@dataclass(frozen=True)
class SubmittedJob:
    phase: str
    job_id: str
    job_name: str
    account: str
    partition: str | None
    time_limit: str
    gpus: int
    cpus: int
    mem: str
    dependency: str | None


def _submit_config_job(
    *,
    phase: str,
    config_path: Path,
    config: dict[str, object],
    nickname_key: str,
    job_prefix: str,
    script_name: str,
    account: str,
    partition: str | None,
    request_gpu: bool,
    dependency: str | None,
    extra_script_args: list[str] | None = None,
) -> SubmittedJob:
    model_cli_name = _require_str(config, "model_cli_name")
    nickname = _require_str(config, nickname_key)
    num_gpus = _require_positive_int(config, "num_gpus") if request_gpu else 0
    time_limit = (
        _hours_to_slurm_time(_require_positive_number(config, "total_time_limit_hours"))
        if request_gpu
        else CPU_TIME_LIMIT
    )
    cpus_per_task = "32" if request_gpu else CPU_CPUS_PER_TASK
    mem = "32G" if request_gpu else CPU_MEM
    job_name = f"{job_prefix}{model_cli_name}_{nickname}"
    log_prefix = job_prefix.rstrip("_")
    notify_start_msg = f"{job_name} started running."
    notify_end_msg = f"{job_name} finished running."
    cmd = [
        "sbatch",
        "--parsable",
        "--job-name",
        job_name,
        "--account",
        account,
        "--output",
        f"slurm/logs/{log_prefix}_%j.out",
        "--error",
        f"slurm/logs/{log_prefix}_%j.err",
        "--cpus-per-task",
        cpus_per_task,
        "--mem",
        mem,
        "--time",
        time_limit,
    ]
    if dependency:
        cmd.extend(["--dependency", dependency])
    if partition:
        cmd.extend(["--partition", partition])
    if request_gpu:
        cmd.extend(["--gres", f"gpu:nvidia_a100:{num_gpus}"])
    cmd.extend(
        [
            str(REPO_ROOT / "slurm" / script_name),
            str(config_path),
            notify_start_msg,
            notify_end_msg,
        ]
    )
    if extra_script_args:
        cmd.extend(extra_script_args)
    job_id = _submit(cmd)
    return SubmittedJob(
        phase=phase,
        job_id=job_id,
        job_name=job_name,
        account=account,
        partition=partition,
        time_limit=time_limit,
        gpus=num_gpus,
        cpus=int(cpus_per_task),
        mem=mem,
        dependency=dependency,
    )


def _submit_training_judging_job(
    *,
    rollout_config: dict[str, object],
    judging_time: str,
    dependency: str | None,
) -> SubmittedJob:
    model_cli_name = _require_str(rollout_config, "model_cli_name")
    config_nickname_rollout = _require_str(rollout_config, "config_nickname_rollout")
    mount_dir = _require_str(rollout_config, "mount_dir")
    tree_dir = (
        f"{mount_dir}/medium_files/{model_cli_name}/{config_nickname_rollout}"
        "/trees_training_oneshot"
    )
    judgment_jsonl = (
        f"{mount_dir}/medium_files/{model_cli_name}/{config_nickname_rollout}"
        "/tree_judgments_training_oneshot.jsonl"
    )
    output_jsonl = (
        f"{mount_dir}/medium_files/{model_cli_name}/{config_nickname_rollout}"
        "/training_judging_outputs.jsonl"
    )
    cache_dir = (
        f"{mount_dir}/medium_files/{model_cli_name}/{config_nickname_rollout}"
        "/judgment_cache"
    )
    escalation_jsonl = (
        f"{mount_dir}/small_files/{model_cli_name}/{config_nickname_rollout}"
        "/judgment_escalations.jsonl"
    )
    job_name = f"judging_{model_cli_name}_{config_nickname_rollout}"
    cmd = [
        "sbatch",
        "--parsable",
        "--job-name",
        job_name,
        "--account",
        "bfsl-delta-cpu",
        "--partition",
        "cpu",
        "--cpus-per-task",
        CPU_CPUS_PER_TASK,
        "--mem",
        CPU_MEM,
        "--time",
        judging_time,
        "--output",
        "slurm/logs/judging_%j.out",
        "--error",
        "slurm/logs/judging_%j.err",
    ]
    if dependency:
        cmd.extend(["--dependency", dependency])
    cmd.extend(
        [
            str(REPO_ROOT / "slurm" / "oneshot_judging_cpu.slurm"),
            "--input-tree-msgpack",
            tree_dir,
            "--output-jsonl",
            output_jsonl,
            "--output-tree-judgment-jsonl",
            judgment_jsonl,
            "--cache-dir",
            cache_dir,
            "--escalation-jsonl",
            escalation_jsonl,
            "--model-cli-name",
            model_cli_name,
            "--dataset-split",
            "training",
            "--notify-start-message",
            f"{job_name} started running.",
            "--notify-end-message",
            f"{job_name} finished running.",
        ]
    )
    job_id = _submit(cmd)
    return SubmittedJob(
        phase="training_judging",
        job_id=job_id,
        job_name=job_name,
        account="bfsl-delta-cpu",
        partition="cpu",
        time_limit=judging_time,
        gpus=0,
        cpus=int(CPU_CPUS_PER_TASK),
        mem=CPU_MEM,
        dependency=dependency,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-config", required=True)
    parser.add_argument("--generation-config", required=True)
    parser.add_argument("--training-config", required=True)
    parser.add_argument(
        "--validation-config",
        default=None,
        help="Defaults to --training-config because current validation configs share the training TOML.",
    )
    parser.add_argument("--judging-time", default=CPU_TIME_LIMIT)
    parser.add_argument(
        "--branching-policy",
        default=None,
        help="Optional bin_oneshot_rollout --branching-policy override.",
    )
    parser.add_argument(
        "--training-advantage-policy",
        default=None,
        help="Optional bin_oneshot_generation --training-advantage-policy override.",
    )
    parser.add_argument(
        "--include-training-chunk-validation",
        action="store_true",
        help="Schedule diagnostic training-chunk validation jobs. Future serious runs leave this off by default.",
    )
    parser.add_argument(
        "--validation-epoch-interval",
        type=int,
        default=10,
        help="Validate held-out epoch 0 and trained epochs divisible by this interval.",
    )
    parser.add_argument(
        "--validation-num-rollout-trials",
        type=int,
        default=None,
        help="Override validation_num_rollout_trials from the validation config.",
    )
    parser.add_argument(
        "--login-smoke",
        action="store_true",
        help="Run login-node smoke checks for all composed binaries and submit nothing.",
    )
    args = parser.parse_args()
    if args.validation_epoch_interval <= 0:
        parser.error("--validation-epoch-interval must be positive")
    if args.validation_num_rollout_trials is not None and args.validation_num_rollout_trials <= 0:
        parser.error("--validation-num-rollout-trials must be positive")

    rollout_path = _resolve_config(args.rollout_config)
    generation_path = _resolve_config(args.generation_config)
    training_path = _resolve_config(args.training_config)
    validation_path = _resolve_config(args.validation_config or args.training_config)
    rollout_config = _read_toml(rollout_path)
    generation_config = _read_toml(generation_path)
    training_config = _read_toml(training_path)
    validation_config = _read_toml(validation_path)
    rollout_extra_args = (
        ["--branching-policy", args.branching_policy] if args.branching_policy else []
    )
    generation_extra_args = (
        ["--training-advantage-policy", args.training_advantage_policy]
        if args.training_advantage_policy
        else []
    )

    if args.login_smoke:
        _run_login_smoke(
            _cargo_smoke(
                "bin_oneshot_rollout",
                "--config-path",
                str(rollout_path),
                *rollout_extra_args,
            )
        )
        model_cli_name = _require_str(rollout_config, "model_cli_name")
        config_nickname_rollout = _require_str(rollout_config, "config_nickname_rollout")
        mount_dir = _require_str(rollout_config, "mount_dir")
        tree_dir = (
            f"{mount_dir}/medium_files/{model_cli_name}/{config_nickname_rollout}"
            "/trees_training_oneshot"
        )
        judgment_jsonl = (
            f"{mount_dir}/medium_files/{model_cli_name}/{config_nickname_rollout}"
            "/tree_judgments_training_oneshot.jsonl"
        )
        output_jsonl = (
            f"{mount_dir}/medium_files/{model_cli_name}/{config_nickname_rollout}"
            "/training_judging_outputs.jsonl"
        )
        cache_dir = (
            f"{mount_dir}/medium_files/{model_cli_name}/{config_nickname_rollout}"
            "/judgment_cache"
        )
        escalation_jsonl = (
            f"{mount_dir}/small_files/{model_cli_name}/{config_nickname_rollout}"
            "/judgment_escalations.jsonl"
        )
        _run_login_smoke(
            _cargo_smoke(
                "bin_oneshot_judging",
                "--input-tree-msgpack",
                tree_dir,
                "--output-jsonl",
                output_jsonl,
                "--output-tree-judgment-jsonl",
                judgment_jsonl,
                "--cache-dir",
                cache_dir,
                "--escalation-jsonl",
                escalation_jsonl,
                "--model-cli-name",
                model_cli_name,
                "--dataset-split",
                "training",
            )
        )
        _run_login_smoke(
            _cargo_smoke(
                "bin_oneshot_generation",
                "--config-path",
                str(generation_path),
                *generation_extra_args,
            )
        )
        if _kl_beta(training_config) > 0.0:
            _run_login_smoke(
                _cargo_smoke("bin_oneshot_ref_logprobs", "--config-path", str(training_path))
            )
        _run_login_smoke(_cargo_smoke("bin_oneshot_training", "--config-path", str(training_path)))
        if args.include_training_chunk_validation:
            _run_login_smoke(
                _cargo_smoke(
                    "bin_oneshot_training_chunk_validation",
                    "--config-path",
                    str(training_path),
                    "--all-chunks",
                    "--phase",
                    "rollout",
                )
            )
        _run_login_smoke(
            _cargo_smoke(
                "bin_oneshot_validation",
                "--config-path",
                str(validation_path),
                "--phase",
                "rollout",
                "--epoch-interval",
                str(args.validation_epoch_interval),
                *(
                    ["--num-rollout-trials", str(args.validation_num_rollout_trials)]
                    if args.validation_num_rollout_trials is not None
                    else []
                ),
            )
        )
        print("All login-smoke checks passed; no jobs submitted.")
        return 0

    jobs: list[SubmittedJob] = []
    rollout = _submit_config_job(
        phase="rollout",
        config_path=rollout_path,
        config=rollout_config,
        nickname_key="config_nickname_rollout",
        job_prefix="rollout_",
        script_name="oneshot_rollout.slurm",
        account="bfsl-delta-gpu",
        partition=None,
        request_gpu=True,
        dependency=None,
        extra_script_args=rollout_extra_args,
    )
    jobs.append(rollout)
    judging = _submit_training_judging_job(
        rollout_config=rollout_config,
        judging_time=args.judging_time,
        dependency=f"afterok:{rollout.job_id}",
    )
    jobs.append(judging)
    generation = _submit_config_job(
        phase="generation",
        config_path=generation_path,
        config=generation_config,
        nickname_key="config_nickname_generation",
        job_prefix="generation_",
        script_name="oneshot_generation_cpu.slurm",
        account="bfsl-delta-cpu",
        partition="cpu",
        request_gpu=False,
        dependency=f"afterok:{judging.job_id}",
        extra_script_args=generation_extra_args,
    )
    jobs.append(generation)
    training_dependency_job_id = generation.job_id
    if _kl_beta(training_config) > 0.0:
        ref_logprobs = _submit_config_job(
            phase="ref_logprobs",
            config_path=training_path,
            config=training_config,
            nickname_key="config_nickname_generation",
            job_prefix="ref_logprobs_",
            script_name="oneshot_ref_logprobs.slurm",
            account="bfsl-delta-gpu",
            partition=None,
            request_gpu=True,
            dependency=f"afterok:{generation.job_id}",
            extra_script_args=None,
        )
        jobs.append(ref_logprobs)
        training_dependency_job_id = ref_logprobs.job_id
    training = _submit_config_job(
        phase="training",
        config_path=training_path,
        config=training_config,
        nickname_key="config_nickname_training",
        job_prefix="training_",
        script_name="oneshot_training.slurm",
        account="bfsl-delta-gpu",
        partition=None,
        request_gpu=True,
        dependency=f"afterok:{training_dependency_job_id}",
        extra_script_args=None,
    )
    jobs.append(training)
    training_chunk_score_job_ids: list[str] = []
    if args.include_training_chunk_validation:
        _require_positive_int(training_config, "num_oneshot_epochs")
        chunk_rollout = _submit_config_job(
            phase="training_chunk_rollout",
            config_path=training_path,
            config=training_config,
            nickname_key="config_nickname_training",
            job_prefix="chunkval_rollout_",
            script_name="oneshot_training_chunk_validation.slurm",
            account="bfsl-delta-gpu",
            partition=None,
            request_gpu=True,
            dependency=f"afterok:{training.job_id}",
            extra_script_args=["rollout"],
        )
        jobs.append(chunk_rollout)
        chunk_judge = _submit_config_job(
            phase="training_chunk_judge",
            config_path=training_path,
            config=training_config,
            nickname_key="config_nickname_training",
            job_prefix="chunkval_judge_",
            script_name="oneshot_training_chunk_validation.slurm",
            account="bfsl-delta-cpu",
            partition="cpu",
            request_gpu=False,
            dependency=f"afterok:{chunk_rollout.job_id}",
            extra_script_args=["judge"],
        )
        jobs.append(chunk_judge)
        chunk_score = _submit_config_job(
            phase="training_chunk_score",
            config_path=training_path,
            config=training_config,
            nickname_key="config_nickname_training",
            job_prefix="chunkval_score_",
            script_name="oneshot_training_chunk_validation.slurm",
            account="bfsl-delta-cpu",
            partition="cpu",
            request_gpu=False,
            dependency=f"afterok:{chunk_judge.job_id}",
            extra_script_args=["score"],
        )
        jobs.append(chunk_score)
        training_chunk_score_job_ids.append(chunk_score.job_id)
    validation_rollout = _submit_config_job(
        phase="validation_rollout",
        config_path=validation_path,
        config=validation_config,
        nickname_key="config_nickname_training",
        job_prefix="validation_rollout_",
        script_name="oneshot_validation.slurm",
        account="bfsl-delta-gpu",
        partition=None,
        request_gpu=True,
        dependency=f"afterok:{training.job_id}",
        extra_script_args=[
            "--phase",
            "rollout",
            "--epoch-interval",
            str(args.validation_epoch_interval),
            *(
                ["--num-rollout-trials", str(args.validation_num_rollout_trials)]
                if args.validation_num_rollout_trials is not None
                else []
            ),
        ],
    )
    jobs.append(validation_rollout)
    validation_judge = _submit_config_job(
        phase="validation_judge",
        config_path=validation_path,
        config=validation_config,
        nickname_key="config_nickname_training",
        job_prefix="validation_judge_",
        script_name="oneshot_validation.slurm",
        account="bfsl-delta-cpu",
        partition="cpu",
        request_gpu=False,
        dependency=f"afterok:{validation_rollout.job_id}",
        extra_script_args=[
            "--phase",
            "judge",
            "--epoch-interval",
            str(args.validation_epoch_interval),
            *(
                ["--num-rollout-trials", str(args.validation_num_rollout_trials)]
                if args.validation_num_rollout_trials is not None
                else []
            ),
        ],
    )
    jobs.append(validation_judge)
    validation_score = _submit_config_job(
        phase="validation_score",
        config_path=validation_path,
        config=validation_config,
        nickname_key="config_nickname_training",
        job_prefix="validation_score_",
        script_name="oneshot_validation.slurm",
        account="bfsl-delta-cpu",
        partition="cpu",
        request_gpu=False,
        dependency=f"afterok:{validation_judge.job_id}",
        extra_script_args=[
            "--phase",
            "score",
            "--epoch-interval",
            str(args.validation_epoch_interval),
            *(
                ["--num-rollout-trials", str(args.validation_num_rollout_trials)]
                if args.validation_num_rollout_trials is not None
                else []
            ),
        ],
    )
    jobs.append(validation_score)
    print("\nSubmitted pipeline jobs:")
    for job in jobs:
        print(
            f"  {job.phase}: id={job.job_id} name={job.job_name} account={job.account} "
            f"partition={job.partition or 'default'} time={job.time_limit} "
            f"gpus={job.gpus} cpus={job.cpus} mem={job.mem} "
            f"dependency={job.dependency or 'none'}"
        )
    print("\nVerify resources with:")
    print("  scontrol show job " + ",".join(job.job_id for job in jobs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
