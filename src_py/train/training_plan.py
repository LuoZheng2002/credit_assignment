from __future__ import annotations

USE_LORA = "lora"
USE_FULL = "full"

DIST_STRATEGY_SINGLE_GPU = "single_gpu"
DIST_STRATEGY_DDP = "ddp"
DIST_STRATEGY_FSDP = "fsdp"


def assert_supported_lora_or_full(value: str) -> str:
    normalized = value.strip()
    assert normalized in {USE_LORA, USE_FULL}, (
        "lora_or_full must be one of: lora, full"
    )
    return normalized


def assert_supported_distributed_strategy(value: str) -> str:
    normalized = value.strip()
    assert normalized in {
        DIST_STRATEGY_SINGLE_GPU,
        DIST_STRATEGY_DDP,
        DIST_STRATEGY_FSDP,
    }, "distributed_strategy must be one of: single_gpu, ddp, fsdp"
    return normalized
