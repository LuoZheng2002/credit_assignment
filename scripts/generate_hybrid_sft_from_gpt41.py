from __future__ import annotations

from _bootstrap import REPO_ROOT
from sft_dataset_generation_common import (
    DEEPSEEK_V4_FLASH_OPENROUTER_MODEL,
    GENERATION_KIND_OPENAI,
    JUDGE_KIND_OPENROUTER,
    OPENAI_CHAT_COMPLETIONS_URL,
    OPENROUTER_CHAT_COMPLETIONS_URL,
    GenerationBackendConfig,
    JudgeBackendConfig,
    ProgramDefaults,
    run_generation_program,
)

DEFAULTS = ProgramDefaults(
    description=(
        "Generate a non-tool-calling SFT dataset by sampling GPT-4.1 trajectories and "
        "keeping only rows judged correct by DeepSeek v4 Flash."
    ),
    generation_backend=GenerationBackendConfig(
        kind=GENERATION_KIND_OPENAI,
        api_url=OPENAI_CHAT_COMPLETIONS_URL,
        default_model="gpt-4.1",
        api_key_env="OPENAI_API_KEY",
        description_label="OpenAI Chat Completions",
    ),
    judge_backend=JudgeBackendConfig(
        kind=JUDGE_KIND_OPENROUTER,
        api_url=OPENROUTER_CHAT_COMPLETIONS_URL,
        model=DEEPSEEK_V4_FLASH_OPENROUTER_MODEL,
        api_key_env="OPENROUTER_API_KEY",
        description_label="OpenRouter DeepSeek v4 Flash",
    ),
    default_output=REPO_ROOT / "datasets" / "hybrid_sft.jsonl",
    default_rejected_output=REPO_ROOT / "datasets" / "hybrid_sft_rejected.jsonl",
    default_progress_path=REPO_ROOT / "datasets" / "hybrid_sft_progress.json",
)


if __name__ == "__main__":
    run_generation_program(DEFAULTS)
