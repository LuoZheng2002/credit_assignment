from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path
from typing import Any

from .engine import TrainConfig, train_with_deepspeed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train causal LM from JSON config")
    parser.add_argument("--config-json-path", type=str, required=True)
    return parser


def _load_train_config_from_json(config_json_path: str) -> TrainConfig:
    config_path = Path(config_json_path)
    assert config_path.exists(), f"config file not found: {config_path}"

    payload: Any = json.loads(config_path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict), "config json root must be an object"

    expected_keys = {field.name for field in fields(TrainConfig)}
    actual_keys = set(payload.keys())
    missing = expected_keys - actual_keys
    extra = actual_keys - expected_keys
    assert len(missing) == 0, f"config missing keys: {sorted(missing)}"
    assert len(extra) == 0, f"config has unknown keys: {sorted(extra)}"

    return TrainConfig(**payload)


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    config = _load_train_config_from_json(config_json_path=args.config_json_path)
    train_with_deepspeed(config)


if __name__ == "__main__":
    main()
