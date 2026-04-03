"""Load a small DeepMath-103K subset for quick access."""

from __future__ import annotations

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
    """Download the first DeepMath-103K samples and persist them locally."""

    repo_root = Path(__file__).resolve().parents[1]
    _install_hf_token(repo_root)

    output_dir = repo_root / "datasets/deepmath_samples"
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset("zwhe99/DeepMath-103K", split="train")
    assert dataset.num_rows >= SAMPLES_TO_SAVE, f"DeepMath-103K must have at least {SAMPLES_TO_SAVE} samples"

    rng = random.Random(SAMPLE_SEED)
    sample_indices = rng.sample(range(dataset.num_rows), SAMPLES_TO_SAVE)
    subset = dataset.select(sample_indices)

    output_file = output_dir / f"DeepMath-103K-first-{SAMPLES_TO_SAVE}.jsonl"
    question_answer_file = output_dir / f"{output_file.stem}-question_answer.jsonl"
    with output_file.open("w", encoding="utf-8") as full_handle, question_answer_file.open(
        "w", encoding="utf-8"
    ) as question_handle:
        for entry in subset:
            json.dump(entry, full_handle, ensure_ascii=False)
            full_handle.write("\n")

            json.dump(
                {"question": entry["question"], "final_answer": entry["final_answer"]},
                question_handle,
                ensure_ascii=False,
            )
            question_handle.write("\n")

    print(f"Saved {SAMPLES_TO_SAVE} DeepMath samples to {output_file.resolve()}")
    print(f"Saved {SAMPLES_TO_SAVE} question-answer entries to {question_answer_file.resolve()}")


if __name__ == "__main__":
    main()
