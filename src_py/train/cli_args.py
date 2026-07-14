from __future__ import annotations

import argparse
from pathlib import Path
from types import UnionType
from typing import Annotated, Any, Literal, TypeVar, Union, get_args, get_origin

from pydantic import BaseModel, ConfigDict, Field


class TrainingHyperparametersRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    lora_or_full: str
    distributed_strategy: str
    advantage_clip: float
    learning_rate: float
    weight_decay: float
    grad_accum_steps: int
    log_time_interval: float
    seed: int
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    lr_warmup_steps: int
    lora_rank: int | None = None
    lora_alpha: int | None = None
    lora_dropout: float | None = None


class TrainingModeOrchestration(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["orchestration"]
    epoch: int
    training_time: float
    input_model_parent_dir: str
    output_model_parent_dir: str
    training_summary_dir: str


class TrainingModeOneShot(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["oneshot"]
    per_epoch_training_time: float
    num_oneshot_epochs: int
    model_output_root: str
    training_summary_dir: str
    base_model_parent_dir: str


TrainingMode = Annotated[
    Union[TrainingModeOrchestration, TrainingModeOneShot],
    Field(discriminator="type"),
]


class TrainingRequestArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    hyperparameters: TrainingHyperparametersRequest
    num_iterations_limit: int
    model_cli_name: str
    config_nickname: str
    hpc_training_root_dir: str | None = None
    training_mode: TrainingMode

    @property
    def epoch(self) -> int:
        if isinstance(self.training_mode, TrainingModeOrchestration):
            return self.training_mode.epoch
        return 0

    @property
    def model_parent_dir(self) -> str:
        if isinstance(self.training_mode, TrainingModeOrchestration):
            return self.training_mode.input_model_parent_dir
        return self.training_mode.base_model_parent_dir

    @property
    def checkpoints_parent_dir(self) -> str:
        if isinstance(self.training_mode, TrainingModeOrchestration):
            return self.training_mode.output_model_parent_dir
        return self.training_mode.model_output_root

    @property
    def final_model_output_parent_dir(self) -> str:
        if isinstance(self.training_mode, TrainingModeOrchestration):
            return self.training_mode.output_model_parent_dir
        return self.training_mode.model_output_root


class TrainProcessLaunchArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    training_trajectory_path: str
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
