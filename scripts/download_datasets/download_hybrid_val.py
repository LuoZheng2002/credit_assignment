"""Build `hybrid_val.jsonl` from DeepMath, MATH, and MetaMathQA."""

from __future__ import annotations

import argparse
from pathlib import Path

from hybrid_jsonl_common import (
    DEFAULT_TRAIN_SAMPLES_PER_DATASET,
    DEFAULT_VAL_SAMPLES_PER_DATASET,
    IN_DISTRIBUTION_DATASET_ORDER,
    assert_enough_rows,
    build_dataset_entries,
    install_hf_token,
    interleave_rows,
    load_in_distribution_datasets,
    write_jsonl_entries,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and build hybrid_val.jsonl")
    parser.add_argument(
        "--samples-per-dataset",
        type=int,
        default=DEFAULT_VAL_SAMPLES_PER_DATASET,
        help=(
            "Number of in-distribution validation samples per dataset "
            f"(default: {DEFAULT_VAL_SAMPLES_PER_DATASET})"
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
        "--output",
        type=Path,
        default=None,
        help="Output JSONL path (default: <repo>/datasets/hybrid_val.jsonl)",
    )
    args = parser.parse_args()

    assert args.samples_per_dataset > 0, "--samples-per-dataset must be positive"
    assert args.train_samples_per_dataset >= 0, (
        "--train-samples-per-dataset must be non-negative"
    )

    repo_root = Path(__file__).resolve().parents[2]
    install_hf_token(repo_root)

    output_path = args.output or (repo_root / "datasets" / "hybrid_val.jsonl")
    datasets = load_in_distribution_datasets()

    rows_by_dataset: dict[str, list[dict[str, object]]] = {}
    for dataset_name in IN_DISTRIBUTION_DATASET_ORDER:
        dataset = datasets[dataset_name]
        assert_enough_rows(
            dataset_name,
            dataset.num_rows,
            args.train_samples_per_dataset,
            args.samples_per_dataset,
        )
        rows_by_dataset[dataset_name] = build_dataset_entries(
            dataset_name,
            dataset,
            start_question_id=args.train_samples_per_dataset,
            target_count=args.samples_per_dataset,
        )

    ordered_rows = interleave_rows(
        rows_by_dataset,
        IN_DISTRIBUTION_DATASET_ORDER,
        rows_per_dataset=args.samples_per_dataset,
    )
    write_jsonl_entries(output_path, ordered_rows)
    print(f"Wrote {len(ordered_rows)} rows to {output_path.resolve()}")


if __name__ == "__main__":
    main()
