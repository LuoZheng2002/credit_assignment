from __future__ import annotations

import argparse
import tomllib
from dataclasses import fields
from pathlib import Path
from typing import Any

from .engine import TrainConfig, train


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train causal LM from TOML config")
    parser.add_argument("--config-toml-path", type=str, required=True)
    return parser


def _load_train_config_from_toml(config_toml_path: str) -> TrainConfig:
    config_path = Path(config_toml_path)
    assert config_path.exists(), f"config file not found: {config_path}"

    payload: Any
    with config_path.open("rb") as handle:
        payload = tomllib.load(handle)
    assert isinstance(payload, dict), "config toml root must be a table"

    expected_keys = {field.name for field in fields(TrainConfig)}
    actual_keys = set(payload.keys())
    if "first_n_training_samples" not in actual_keys:
        payload["first_n_training_samples"] = 0
        actual_keys.add("first_n_training_samples")
    missing = expected_keys - actual_keys
    extra = actual_keys - expected_keys
    assert len(missing) == 0, f"config missing keys: {sorted(missing)}"
    assert len(extra) == 0, f"config has unknown keys: {sorted(extra)}"

    return TrainConfig(**payload)


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    config = _load_train_config_from_toml(config_toml_path=args.config_toml_path)
    train(config)


if __name__ == "__main__":
    main()
