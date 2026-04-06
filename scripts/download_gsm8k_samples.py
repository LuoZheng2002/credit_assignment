"""Load a small GSM8K subset for quick access."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

from datasets import load_dataset
from dotenv import load_dotenv

SAMPLES_TO_SAVE = 10
SAMPLE_SEED = 2026


def _install_hf_token(repo_root: Path) -> None:
    """Ensure HF_TOKEN is available in the environment before downloading."""

    env_file = repo_root / ".env"
    assert env_file.exists(), "Missing .env file; cannot load HF_TOKEN"
    load_dotenv(env_file, override=True)
    assert os.environ.get("HF_TOKEN"), "HF_TOKEN not defined in .env"


def main() -> None:
    """Download a subset of GSM8K and persist it locally."""

    repo_root = Path(__file__).resolve().parents[1]
    _install_hf_token(repo_root)

    parser = argparse.ArgumentParser(description="Download GSM8K samples")
    parser.add_argument(
        "--num-samples",
        type=int,
        default=SAMPLES_TO_SAVE,
        help="Number of samples to download (must be positive)",
    )
    args = parser.parse_args()
    num_samples = args.num_samples
    assert num_samples > 0, "--num-samples must be positive"

    output_dir = repo_root / "datasets"
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset("gsm8k", "main", split="train")
    assert dataset.num_rows >= num_samples, f"GSM8K train split must have at least {num_samples} samples"

    rng = random.Random(SAMPLE_SEED)
    sample_indices = rng.sample(range(dataset.num_rows), num_samples)
    subset = dataset.select(sample_indices)

    question_answer_file = output_dir / f"gsm8k_samples_{num_samples}.jsonl"
    question_answer_reasoning_file = output_dir / f"gsm8k_samples_{num_samples}_reasoning.jsonl"
    with question_answer_file.open("w", encoding="utf-8") as question_handle, question_answer_reasoning_file.open(
        "w", encoding="utf-8"
    ) as reasoning_handle:
        for idx, entry in enumerate(subset):
            assert "question" in entry, "Expected GSM8K entry to contain question"
            assert "answer" in entry, "Expected GSM8K entry to contain answer"
            answer_parts = entry["answer"].split("#### ", 1)
            assert len(answer_parts) == 2, "GSM8K answer field must contain '#### ' separator"
            reasoning_part, final_answer_part = answer_parts
            assert reasoning_part, "Reasoning part of GSM8K answer must not be empty"
            assert final_answer_part, "Final answer part of GSM8K answer must not be empty"

            json.dump(
                {"id": idx, "question": entry["question"], "final_answer": final_answer_part},
                question_handle,
                ensure_ascii=False,
            )
            question_handle.write("\n")
            json.dump(
                {
                    "id": idx,
                    "question": entry["question"],
                    "final_answer": final_answer_part,
                    "reasoning": reasoning_part,
                },
                reasoning_handle,
                ensure_ascii=False,
            )
            reasoning_handle.write("\n")

    print(f"Saved {num_samples} GSM8K samples to {question_answer_file.resolve()}")
    print(
        f"Saved {num_samples} GSM8K samples with reasoning to {question_answer_reasoning_file.resolve()}"
    )


if __name__ == "__main__":
    main()
