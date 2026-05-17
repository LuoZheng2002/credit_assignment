"""Build a hybrid training sqlite database from DeepMath, MATH, and GSM8K.

The output schema matches `research_utility::sqlite_store::SqliteStore`:
- table name: store_entries
- columns: id TEXT PRIMARY KEY, payload_json TEXT NOT NULL

Each payload is a JSON object matching `HybridDatasetEntry`.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
from pathlib import Path

from datasets import concatenate_datasets, load_dataset
from dotenv import load_dotenv

STORE_TABLE_NAME = "store_entries"
SAMPLES_PER_DATASET = 5_000


def _install_hf_token(repo_root: Path) -> None:
    """Ensure HF_TOKEN is available before downloading datasets."""

    env_file = repo_root / ".env"
    assert env_file.exists(), "Missing .env file; cannot load HF_TOKEN"
    load_dotenv(env_file, override=True)
    assert os.environ.get("HF_TOKEN"), "HF_TOKEN not defined in .env"


def _extract_boxed_content(text: str) -> str | None:
    marker = "\\boxed{"
    start = text.find(marker)
    if start < 0:
        return None

    depth = 1
    content_chars: list[str] = []
    for ch in text[start + len(marker) :]:
        if ch == "{":
            depth += 1
            content_chars.append(ch)
        elif ch == "}":
            depth -= 1
            if depth == 0:
                content = "".join(content_chars).strip()
                return content or None
            content_chars.append(ch)
        else:
            content_chars.append(ch)
    return None


def _parse_gsm8k_final_answer(answer: str) -> str:
    parts = answer.split("####", 1)
    if len(parts) == 2:
        return parts[1].strip()
    return answer.strip()


def _get_required_field(raw: dict[str, object], dataset_name: str, field_candidates: list[str]) -> str:
    for field_name in field_candidates:
        if field_name in raw and raw[field_name] is not None:
            value = raw[field_name]
            if isinstance(value, str):
                return value
            return str(value)

    available = ", ".join(sorted(raw.keys()))
    expected = ", ".join(field_candidates)
    raise AssertionError(
        f"{dataset_name} entry missing required field(s): {expected}. Available fields: {available}"
    )


def _load_math_train() -> tuple[object, bool]:
    """Load MATH train split.

    Returns (dataset, used_repetition).
    """

    try:
        dataset = load_dataset("EleutherAI/hendrycks_math", "all", split="train")
    except Exception:
        configs = [
            "algebra",
            "counting_and_probability",
            "geometry",
            "intermediate_algebra",
            "number_theory",
            "prealgebra",
            "precalculus",
        ]
        parts = [load_dataset("EleutherAI/hendrycks_math", cfg, split="train") for cfg in configs]
        dataset = concatenate_datasets(parts)

    assert dataset.num_rows > 0, "MATH train split must be non-empty"
    used_repetition = dataset.num_rows < SAMPLES_PER_DATASET
    return dataset, used_repetition


def _build_dataset_entries(dataset_name: str, dataset: object, target_count: int) -> list[dict[str, object]]:
    assert target_count > 0, "target_count must be positive"
    num_rows = dataset.num_rows
    assert num_rows > 0, f"{dataset_name} must contain at least one row"

    rows: list[dict[str, object]] = []
    for i in range(target_count):
        question_id = i % num_rows
        raw = dataset[question_id]

        if dataset_name == "deepmath":
            question = _get_required_field(raw, dataset_name, ["question"])
            correct_answer = _get_required_field(raw, dataset_name, ["final_answer", "answer"])
        elif dataset_name == "math":
            question = _get_required_field(raw, dataset_name, ["problem"])
            solution = _get_required_field(raw, dataset_name, ["solution"])
            correct_answer = _extract_boxed_content(solution) or solution.strip()
        elif dataset_name == "gsm8k":
            question = _get_required_field(raw, dataset_name, ["question"])
            answer = _get_required_field(raw, dataset_name, ["answer"])
            correct_answer = _parse_gsm8k_final_answer(answer)
        else:
            raise AssertionError(f"Unknown dataset_name: {dataset_name}")

        rows.append(
            {
                "dataset_name": dataset_name,
                "question_id": question_id,
                "question": question,
                "correct_answer": correct_answer,
            }
        )
    return rows


def _write_store_entries(db_path: Path, payload_rows: list[dict[str, object]]) -> None:
    if db_path.exists():
        db_path.unlink()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(db_path)
    try:
        with connection:
            connection.execute(
                f"""
                CREATE TABLE {STORE_TABLE_NAME} (
                    id TEXT PRIMARY KEY,
                    payload_json TEXT NOT NULL
                )
                """
            )

            for row in payload_rows:
                flat_id = row["flat_id"]
                payload_json = json.dumps(row, ensure_ascii=False)
                connection.execute(
                    f"INSERT INTO {STORE_TABLE_NAME} (id, payload_json) VALUES (?, ?)",
                    (str(flat_id), payload_json),
                )
    finally:
        connection.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and build hybrid_train.sqlite")
    parser.add_argument(
        "--samples-per-dataset",
        type=int,
        default=SAMPLES_PER_DATASET,
        help=f"Number of samples per dataset (default: {SAMPLES_PER_DATASET})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output sqlite path (default: <repo>/datasets/hybrid_train.sqlite)",
    )
    args = parser.parse_args()

    assert args.samples_per_dataset > 0, "--samples-per-dataset must be positive"

    repo_root = Path(__file__).resolve().parents[2]
    _install_hf_token(repo_root)

    output_path = args.output or (repo_root / "datasets" / "hybrid_train.sqlite")

    deepmath = load_dataset("mlfoundations-dev/deepmath", split="train")
    math, math_repeated = _load_math_train()
    gsm8k = load_dataset("openai/gsm8k", "main", split="train")

    deepmath_rows = _build_dataset_entries("deepmath", deepmath, args.samples_per_dataset)
    math_rows = _build_dataset_entries("math", math, args.samples_per_dataset)
    gsm8k_rows = _build_dataset_entries("gsm8k", gsm8k, args.samples_per_dataset)

    ordered_rows: list[dict[str, object]] = []
    interleave_order = ("deepmath", "math", "gsm8k")
    per_dataset = args.samples_per_dataset

    for i in range(per_dataset):
        for dataset_name in interleave_order:
            source = {
                "deepmath": deepmath_rows,
                "math": math_rows,
                "gsm8k": gsm8k_rows,
            }[dataset_name][i]

            flat_id = len(ordered_rows)
            ordered_rows.append(
                {
                    "flat_id": flat_id,
                    "dataset_name": source["dataset_name"],
                    "question_id": source["question_id"],
                    "question": source["question"],
                    "correct_answer": source["correct_answer"],
                }
            )

    _write_store_entries(output_path, ordered_rows)

    print(f"Wrote {len(ordered_rows)} rows to {output_path.resolve()}")
    if math_repeated:
        print(
            "Note: MATH train split has fewer than requested rows; "
            "rows were repeated from the start to reach the target count."
        )


if __name__ == "__main__":
    main()
