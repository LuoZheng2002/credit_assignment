"""Build an aggregated test sqlite database from 5 datasets.

Datasets are concatenated in order: deepmath, math, gsm8k, aime25, amc23.
The test set does not overlap with hybrid_train.sqlite or hybrid_val.sqlite.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from research_utility import SqliteStore
from test_set_sqlite_common import (
    TRAINING_DATASET_NAMES,
    install_hf_token,
    load_aime25_test,
    load_amc23_test,
    load_deepmath_test,
    load_gsm8k_test,
    load_math_test,
    normalize_entry,
    _sample_question_ids,
)

TRAIN_SAMPLES_PER_DATASET = 5_000
VAL_SAMPLES_PER_DATASET = 1_000
COMBINED_CLEARANCE = TRAIN_SAMPLES_PER_DATASET + VAL_SAMPLES_PER_DATASET
MIN_SAMPLES = 1_000
DEFAULT_SAMPLE_SEED = 42

DATASET_ORDER = ("deepmath", "math", "gsm8k", "aime25", "amc23")

_DATASET_LOADERS = {
    "math": load_math_test,
    "gsm8k": load_gsm8k_test,
    "aime25": load_aime25_test,
    "amc23": load_amc23_test,
}


def _write_store_entries(db_path: Path, payload_rows: list[dict[str, object]]) -> None:
    if db_path.exists():
        db_path.unlink()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    store = SqliteStore[str, dict[str, object]](db_path)
    try:
        for row in payload_rows:
            flat_id = row["flat_id"]
            store.upsert(str(flat_id), row)
    finally:
        store.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and build aggregated_test.sqlite")
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=DEFAULT_SAMPLE_SEED,
        help=f"Random seed for sampling (default: {DEFAULT_SAMPLE_SEED})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output sqlite path (default: <repo>/datasets/aggregated_test.sqlite)",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    install_hf_token(repo_root)

    output_path = args.output or (repo_root / "datasets" / "aggregated_test.sqlite")

    all_rows: list[dict[str, object]] = []
    flat_id = 0
    offsets: dict[str, int] = {}
    counts: dict[str, int] = {}

    for dataset_name in DATASET_ORDER:
        offsets[dataset_name] = flat_id

        if dataset_name == "deepmath":
            dataset, source_split = load_deepmath_test()
        else:
            dataset = _DATASET_LOADERS[dataset_name]()
            source_split = "test"

        num_rows = dataset.num_rows
        target_count = min(num_rows, MIN_SAMPLES)

        clearance_override = None
        if source_split == "train" and dataset_name in TRAINING_DATASET_NAMES:
            clearance_override = COMBINED_CLEARANCE

        question_ids = _sample_question_ids(
            dataset_name,
            num_rows,
            target_count,
            source_split,
            args.sample_seed,
            clearance_override=clearance_override,
        )

        dataset_rows = 0
        for qid in question_ids:
            raw = dataset[qid]
            question, correct_answer = normalize_entry(dataset_name, raw)
            all_rows.append(
                {
                    "flat_id": flat_id,
                    "dataset_name": dataset_name,
                    "question_id": qid,
                    "question": question,
                    "correct_answer": correct_answer,
                }
            )
            flat_id += 1
            dataset_rows += 1

        counts[dataset_name] = dataset_rows

    _write_store_entries(output_path, all_rows)

    print(f"Wrote {len(all_rows)} rows to {output_path.resolve()}")
    print()
    for dataset_name in DATASET_ORDER:
        print(f"  {dataset_name}: {counts[dataset_name]} samples (offset={offsets[dataset_name]})")


if __name__ == "__main__":
    main()