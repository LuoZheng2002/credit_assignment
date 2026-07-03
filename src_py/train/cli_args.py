from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import UnionType
from typing import Any, TypeVar, Union, get_args, get_origin

from pydantic import BaseModel, ConfigDict


class TrainingRequestArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    training_plan: str
    advantage_clip: float
    learning_rate: float
    weight_decay: float
    grad_accum_steps: int
    log_time_interval: float
    checkpoint_save_time_interval: float
    seed: int
    training_time: float
    num_iterations_limit: int
    artifact_root_dir: str
    model_cli_name: str
    config_nickname: str
    epoch: int
    model_parent_dir: str
    checkpoints_parent_dir: str
    final_model_output_parent_dir: str
    hpc_training_root_dir: str | None = None
    lora_rank: int | None = None
    lora_alpha: int | None = None
    lora_dropout: float | None = None
    lora_target_modules_csv: str | None = None
    resume_checkpoint_tag: str | None = None
    # How negative-advantage tokens contribute to the loss: "weighted_ce"
    # (legacy, unbounded below) or "unlikelihood" (bounded). Mirrors the
    # optional negative_loss_mode field of PythonTrainingConfigCommon.
    negative_loss_mode: str = "weighted_ce"
    # Multiplier on the negative-advantage contribution relative to positive
    # imitation (1.0 = symmetric; unlikelihood at 1.0 hurt validation -4.8pt).
    negative_loss_weight: float = 1.0
    adam_fp32: bool


class SftTrainingRequestArgs(BaseModel):
    """Simplified training request args for SFT (no advantage_clip, advantage-related fields)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    training_plan: str
    learning_rate: float
    weight_decay: float
    grad_accum_steps: int
    log_time_interval: float
    checkpoint_save_time_interval: float
    seed: int
    training_time: float
    num_iterations_limit: int
    artifact_root_dir: str
    model_cli_name: str
    config_nickname: str
    model_parent_dir: str
    checkpoints_parent_dir: str
    final_model_output_parent_dir: str
    hpc_training_root_dir: str | None = None
    lora_rank: int | None = None
    lora_alpha: int | None = None
    lora_dropout: float | None = None
    lora_target_modules_csv: str | None = None
    resume_checkpoint_tag: str | None = None
    adam_fp32: bool


class TrainingWrapperLaunchArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    num_gpus: int
    trajectory_sqlite_path: str
    hf_model_name: str
    wrapper_log_path: str
    orchestrator_socket_path: str = ""
    test_sleep_secs: float = 0.0


class SftWrapperLaunchArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    num_gpus: int
    sft_training_data_path: str
    hf_model_name: str
    wrapper_log_path: str
    orchestrator_socket_path: str = ""


class SftTrainProcessLaunchArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    sft_training_data_path: str
    training_request_json_path: str
    orchestrator_socket_path: str = ""


class TrainProcessLaunchArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    training_trajectory_sqlite_path: str
    training_request_json_path: str
    orchestrator_socket_path: str = ""


T = TypeVar("T", bound=BaseModel)


def add_model_arguments(
    parser: argparse.ArgumentParser,
    model_type: type[BaseModel],
    *,
    exclude: set[str] | None = None,
) -> None:
    excluded = exclude or set()
    for field_name, field_info in model_type.model_fields.items():
        if field_name in excluded:
            continue

        argument_type = _argument_type(field_info.annotation)
        kwargs: dict[str, Any] = {
            "type": _parse_bool
            if argument_type is bool
            else str
            if argument_type is Path
            else argument_type,
        }
        if field_info.is_required():
            kwargs["required"] = True
        else:
            kwargs["default"] = field_info.default

        parser.add_argument(f"--{field_name.replace('_', '-')}", **kwargs)


def parse_model_args(parser: argparse.ArgumentParser, model_type: type[T]) -> T:
    namespace = parser.parse_args()
    return model_type.model_validate(vars(namespace))


def parse_model_stdin(model_type: type[T]) -> T:
    raw = sys.stdin.buffer.read()
    if not raw or not raw.strip():
        raise ValueError(f"expected JSON payload on stdin for {model_type.__name__}")
    return model_type.model_validate_json(raw)


def parse_model_json_file(model_type: type[T], json_path: str | Path) -> T:
    path = Path(json_path)
    raw = path.read_bytes()
    if not raw or not raw.strip():
        raise ValueError(
            f"expected JSON payload in file for {model_type.__name__}: {path}"
        )
    return model_type.model_validate_json(raw)


def model_to_payload(
    instance: BaseModel,
    *,
    exclude: set[str] | None = None,
) -> dict[str, Any]:
    excluded = exclude or set()
    return instance.model_dump(mode="python", exclude_none=True, exclude=excluded)


def model_to_json_bytes(instance: BaseModel) -> bytes:
    return instance.model_dump_json(exclude_none=True).encode("utf-8")


def write_model_json_file(instance: BaseModel, json_path: str | Path) -> Path:
    path = Path(json_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(instance.model_dump_json(exclude_none=True), encoding="utf-8")
    return path


def model_to_cli_args(
    instance: BaseModel,
    *,
    exclude: set[str] | None = None,
) -> list[str]:
    payload = model_to_payload(instance, exclude=exclude)
    args: list[str] = []
    for key in sorted(payload.keys()):
        value = payload[key]
        args.append(f"--{key.replace('_', '-')}")
        if isinstance(value, bool):
            args.append("true" if value else "false")
        else:
            args.append(str(value))
    return args


def _argument_type(annotation: Any) -> type[Any]:
    origin = get_origin(annotation)
    if origin in (Union, UnionType):
        members = [
            member for member in get_args(annotation) if member is not type(None)
        ]
        if len(members) != 1:
            raise TypeError(f"unsupported CLI union annotation: {annotation!r}")
        return _argument_type(members[0])
    if annotation in (str, int, float, bool, Path):
        return annotation
    raise TypeError(f"unsupported CLI field annotation: {annotation!r}")


def _parse_bool(raw: str) -> bool:
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {raw}")
