"""Re-export shim for backward compatibility.

All models and utilities are now defined in
:mod:`src_py.training_config_models` (outside the ``train`` package) so
that the training wrapper can import them without triggering a torch
import via ``src_py/train/__init__.py``.
"""

from __future__ import annotations

from src_py.training_config_models import (  # noqa: F401
    T,
    TrainingHyperparametersRequest,
    TrainingMode,
    TrainingModeOneShot,
    TrainingModeOrchestration,
    TrainingRequestArgs,
    TrainProcessLaunchArgs,
    _argument_type,  # noqa: F401
    _parse_bool,  # noqa: F401
    add_model_arguments,
    model_to_cli_args,
    model_to_json_bytes,
    model_to_payload,
    parse_model_args,
    parse_model_json_file,
    write_model_json_file,
)
