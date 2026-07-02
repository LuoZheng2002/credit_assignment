"""Build `hybrid_val.jsonl` from DeepMath, MATH, and NuminaMath."""

from __future__ import annotations

import argparse
from pathlib import Path

from hybrid_jsonl_common import (
    DEFAULT_VAL_SAMPLES_PER_DATASET,
    IN_DISTRIBUTION_DATASET_ORDER,
    assert_enough_rows,
    build_dataset_entries,
    install_hf_token,
    interleave_rows,
    load_in_distribution_datasets,
    write_jsonl_entries,
)

DEFAULT_MATH_TRAIN = 6_500
DEFAULT_DEEPMATH_TRAIN = 13_000
DEFAULT_NUMINAMATH_TRAIN = 13_000


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and build hybrid_val.jsonl")
    parser.add_argument(
        "--samples-per-dataset",
        type=int,
        default=DEFAULT_VAL_SAMPLES_PER_DATASET,
        help=(
            "Number of validation samples per dataset "
            f"(default: {DEFAULT_VAL_SAMPLES_PER_DATASET})"
        ),
    )
    parser.add_argument(
        "--math-train-samples",
        type=int,
        default=DEFAULT_MATH_TRAIN,
        help=(
            "Number of MATH training samples already reserved "
            f"(default: {DEFAULT_MATH_TRAIN})"
        ),
    )
    parser.add_argument(
        "--deepmath-train-samples",
        type=int,
        default=DEFAULT_DEEPMATH_TRAIN,
        help=(
            "Number of DeepMath training samples already reserved "
            f"(default: {DEFAULT_DEEPMATH_TRAIN})"
        ),
    )
    parser.add_argument(
        "--numinamath-train-samples",
        type=int,
        default=DEFAULT_NUMINAMATH_TRAIN,
        help=(
            "Number of NuminaMath training samples already reserved "
            f"(default: {DEFAULT_NUMINAMATH_TRAIN})"
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
    for label, val in (
        ("--math-train-samples", args.math_train_samples),
        ("--deepmath-train-samples", args.deepmath_train_samples),
        ("--numinamath-train-samples", args.numinamath_train_samples),
    ):
        assert val >= 0, f"{label} must be non-negative"

    repo_root = Path(__file__).resolve().parents[2]
    install_hf_token(repo_root)

    output_path = args.output or (repo_root / "datasets" / "hybrid_val.jsonl")
    datasets = load_in_distribution_datasets()

    train_offsets = {
        "deepmath": args.deepmath_train_samples,
        "math": args.math_train_samples,
        "numinamath": args.numinamath_train_samples,
    }

    rows_by_dataset: dict[str, list[dict[str, object]]] = {}
    for dataset_name in IN_DISTRIBUTION_DATASET_ORDER:
        dataset = datasets[dataset_name]
        offset = train_offsets[dataset_name]
        assert_enough_rows(
            dataset_name,
            dataset.num_rows,
            offset,
            args.samples_per_dataset,
        )
        rows_by_dataset[dataset_name] = build_dataset_entries(
            dataset_name,
            dataset,
            start_question_id=offset,
            target_count=args.samples_per_dataset,
        )

    ordered_rows = interleave_rows(
        rows_by_dataset,
        IN_DISTRIBUTION_DATASET_ORDER,
        rows_per_dataset=args.samples_per_dataset,
    )
    write_jsonl_entries(output_path, ordered_rows)
    print(f"Wrote {len(ordered_rows)} rows to {output_path.resolve()}")
    for name in IN_DISTRIBUTION_DATASET_ORDER:
        offset = train_offsets[name]
        print(f"  {name}: {args.samples_per_dataset} samples (offset={offset})")


if __name__ == "__main__":
    main()
