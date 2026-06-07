from __future__ import annotations

TRAINING_PLAN_LORA = "lora"
TRAINING_PLAN_FSDP = "fsdp"

_LEGACY_PLAN_ALIASES = {
    "lora_current": TRAINING_PLAN_LORA,
    "full_fsdp_backup": TRAINING_PLAN_FSDP,
}


def normalize_training_plan(plan_name: str) -> str:
    normalized = plan_name.strip()
    assert len(normalized) > 0, "training_plan cannot be empty"
    return _LEGACY_PLAN_ALIASES.get(normalized, normalized)


def assert_supported_training_plan(plan_name: str) -> str:
    normalized = normalize_training_plan(plan_name)
    assert normalized in {
        TRAINING_PLAN_LORA,
        TRAINING_PLAN_FSDP,
    }, "training_plan must be one of: lora, fsdp"
    return normalized
