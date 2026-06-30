"""Build `hybrid_test.jsonl` from in-distribution and OOD evaluation datasets.

Dataset order:
- deepmath
- math
- metamathqa
- amc2023
- gaokao_math_2024
- collegemath
- numinamath

For datasets with a published test split, samples come from that test split.
For datasets without a test split, samples come from non-overlapping train rows
that start after the train+validation clearance reserved for the hybrid
in-distribution datasets.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from hybrid_jsonl_common import (
    DEFAULT_SAMPLE_SEED,
    DEFAULT_TEST_SAMPLES_PER_DATASET,
    DEFAULT_TRAIN_SAMPLES_PER_DATASET,
    DEFAULT_VAL_SAMPLES_PER_DATASET,
    EVALUATION_DATASET_ORDER,
    IN_DISTRIBUTION_DATASET_NAMES,
    install_hf_token,
    load_evaluation_dataset,
    normalize_entry,
    sample_question_ids,
    write_jsonl_entries,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and build hybrid_test.jsonl")
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=DEFAULT_SAMPLE_SEED,
        help=f"Random seed for sampling (default: {DEFAULT_SAMPLE_SEED})",
    )
    parser.add_argument(
        "--max-samples-per-dataset",
        type=int,
        default=DEFAULT_TEST_SAMPLES_PER_DATASET,
        help=(
            "Upper bound on evaluation rows per dataset "
            f"(default: {DEFAULT_TEST_SAMPLES_PER_DATASET})"
        ),
    )
    parser.add_argument(
        "--train-samples-per-dataset",
        type=int,
        default=DEFAULT_TRAIN_SAMPLES_PER_DATASET,
        help=(
            "Number of in-distribution training samples already reserved per dataset "
            f"(default: {DEFAULT_TRAIN_SAMPLES_PER_DATASET})"
        ),
    )
    parser.add_argument(
        "--val-samples-per-dataset",
        type=int,
        default=DEFAULT_VAL_SAMPLES_PER_DATASET,
        help=(
            "Number of in-distribution validation samples already reserved per dataset "
            f"(default: {DEFAULT_VAL_SAMPLES_PER_DATASET})"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSONL path (default: <repo>/datasets/hybrid_test.jsonl)",
    )
    args = parser.parse_args()

    assert args.max_samples_per_dataset > 0, (
        "--max-samples-per-dataset must be positive"
    )
    assert args.train_samples_per_dataset >= 0, (
        "--train-samples-per-dataset must be non-negative"
    )
    assert args.val_samples_per_dataset >= 0, (
        "--val-samples-per-dataset must be non-negative"
    )

    repo_root = Path(__file__).resolve().parents[2]
    install_hf_token(repo_root)

    output_path = args.output or (repo_root / "datasets" / "hybrid_test.jsonl")
    combined_clearance = args.train_samples_per_dataset + args.val_samples_per_dataset

    all_rows: list[dict[str, object]] = []
    flat_id = 0
    offsets: dict[str, int] = {}
    counts: dict[str, int] = {}

    for dataset_name in EVALUATION_DATASET_ORDER:
        offsets[dataset_name] = flat_id
        dataset, source_split = load_evaluation_dataset(dataset_name)
        num_rows = dataset.num_rows
        assert num_rows > 0, f"{dataset_name} split must be non-empty"

        start_question_id = 0
        if source_split == "train" and dataset_name in IN_DISTRIBUTION_DATASET_NAMES:
            assert num_rows > combined_clearance, (
                f"{dataset_name} has {num_rows} train rows, but {combined_clearance} are reserved for "
                "hybrid_train and hybrid_val, leaving no non-overlapping rows for hybrid_test"
            )
            start_question_id = combined_clearance
            available_rows = num_rows - start_question_id
        else:
            available_rows = num_rows

        target_count = min(available_rows, args.max_samples_per_dataset)
        question_ids = sample_question_ids(
            start_question_id=start_question_id,
            num_rows=num_rows,
            target_count=target_count,
            sample_seed=args.sample_seed,
        )

        dataset_rows = 0
        for question_id in question_ids:
            raw = dataset[question_id]
            question, correct_answer = normalize_entry(dataset_name, raw)
            all_rows.append(
                {
                    "flat_id": flat_id,
                    "dataset_name": dataset_name,
                    "question_id": question_id,
                    "question": question,
                    "correct_answer": correct_answer,
                }
            )
            flat_id += 1
            dataset_rows += 1

        counts[dataset_name] = dataset_rows

    write_jsonl_entries(output_path, all_rows)

    print(f"Wrote {len(all_rows)} rows to {output_path.resolve()}")
    print()
    for dataset_name in EVALUATION_DATASET_ORDER:
        print(
            f"  {dataset_name}: {counts[dataset_name]} samples (offset={offsets[dataset_name]})"
        )


if __name__ == "__main__":
    main()
