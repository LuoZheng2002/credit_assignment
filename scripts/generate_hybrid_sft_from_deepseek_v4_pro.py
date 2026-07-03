from __future__ import annotations

from _bootstrap import REPO_ROOT
from sft_dataset_generation_common import (
    DEEPSEEK_CHAT_COMPLETIONS_URL,
    DEEPSEEK_V4_FLASH_MODEL,
    GENERATION_KIND_DEEPSEEK_OFFICIAL,
    JUDGE_KIND_DEEPSEEK_OFFICIAL,
    GenerationBackendConfig,
    JudgeBackendConfig,
    ProgramDefaults,
    run_generation_program,
)

DEFAULTS = ProgramDefaults(
    description=(
        "Generate a non-tool-calling SFT dataset by sampling DeepSeek v4 Pro trajectories "
        "through the official DeepSeek API and keeping only rows judged correct by DeepSeek v4 Flash."
    ),
    generation_backend=GenerationBackendConfig(
        kind=GENERATION_KIND_DEEPSEEK_OFFICIAL,
        api_url=DEEPSEEK_CHAT_COMPLETIONS_URL,
        default_model="deepseek-v4-pro",
        api_key_env="DEEPSEEK_API_KEY",
        description_label="DeepSeek official Chat Completions",
    ),
    judge_backend=JudgeBackendConfig(
        kind=JUDGE_KIND_DEEPSEEK_OFFICIAL,
        api_url=DEEPSEEK_CHAT_COMPLETIONS_URL,
        model=DEEPSEEK_V4_FLASH_MODEL,
        api_key_env="DEEPSEEK_API_KEY",
        description_label="DeepSeek official v4 Flash",
    ),
    default_output=REPO_ROOT / "datasets" / "hybrid_sft_deepseek_v4_pro.jsonl",
    default_rejected_output=REPO_ROOT
    / "datasets"
    / "hybrid_sft_deepseek_v4_pro_rejected.jsonl",
    default_progress_path=REPO_ROOT
    / "datasets"
    / "hybrid_sft_deepseek_v4_pro_progress.json",
)


if __name__ == "__main__":
    run_generation_program(DEFAULTS)
