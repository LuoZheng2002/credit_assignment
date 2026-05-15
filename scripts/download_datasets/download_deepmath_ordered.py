"""Load the first N DeepMath-103K samples without shuffling."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from datasets import load_dataset
from dotenv import load_dotenv


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

    parser = argparse.ArgumentParser(description="Download ordered DeepMath samples")
    parser.add_argument(
        "--num-samples",
        type=int,
        required=True,
        help="Number of initial samples to download (must be positive)",
    )
    args = parser.parse_args()
    num_samples = args.num_samples
    assert num_samples > 0, "--num-samples must be positive"

    output_dir = repo_root / "datasets"
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset("zwhe99/DeepMath-103K", split="train")
    assert dataset.num_rows >= num_samples, f"DeepMath-103K must have at least {num_samples} samples"

    subset = dataset.select(range(num_samples))

    output_file = output_dir / f"deepmath_ordered_{num_samples}.jsonl"
    with output_file.open("w", encoding="utf-8") as handle:
        for idx, entry in enumerate(subset):
            json.dump(
                {"id": idx, "question": entry["question"], "final_answer": entry["final_answer"]},
                handle,
                ensure_ascii=False,
            )
            handle.write("\n")

    print(f"Saved {num_samples} ordered DeepMath samples to {output_file.resolve()}")


if __name__ == "__main__":
    main()
