"""Build `hybrid_train.jsonl` from DeepMath, MATH, and NuminaMath with weighted interleaving."""

from __future__ import annotations

import argparse
from pathlib import Path

from hybrid_jsonl_common import (
    IN_DISTRIBUTION_DATASET_ORDER,
    TRAIN_INTERLEAVE_PATTERN,
    assert_enough_rows,
    build_dataset_entries,
    install_hf_token,
    interleave_rows_weighted,
    load_in_distribution_datasets,
    write_jsonl_entries,
)

# Defaults implement a 1:2:2 ratio (Math : DeepMath : NuminaMath).
DEFAULT_MATH_SAMPLES = 6_500
DEFAULT_DEEPMATH_SAMPLES = 13_000
DEFAULT_NUMINAMATH_SAMPLES = 13_000


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and build hybrid_train.jsonl"
    )
    parser.add_argument(
        "--math-samples",
        type=int,
        default=DEFAULT_MATH_SAMPLES,
        help=f"Number of MATH training samples (default: {DEFAULT_MATH_SAMPLES})",
    )
    parser.add_argument(
        "--deepmath-samples",
        type=int,
        default=DEFAULT_DEEPMATH_SAMPLES,
        help=f"Number of DeepMath training samples (default: {DEFAULT_DEEPMATH_SAMPLES})",
    )
    parser.add_argument(
        "--numinamath-samples",
        type=int,
        default=DEFAULT_NUMINAMATH_SAMPLES,
        help=f"Number of NuminaMath training samples (default: {DEFAULT_NUMINAMATH_SAMPLES})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSONL path (default: <repo>/datasets/hybrid_train.jsonl)",
    )
    args = parser.parse_args()

    assert args.math_samples > 0, "--math-samples must be positive"
    assert args.deepmath_samples > 0, "--deepmath-samples must be positive"
    assert args.numinamath_samples > 0, "--numinamath-samples must be positive"

    repo_root = Path(__file__).resolve().parents[2]
    install_hf_token(repo_root)

    output_path = args.output or (repo_root / "datasets" / "hybrid_train.jsonl")
    datasets = load_in_distribution_datasets()

    sample_counts = {
        "deepmath": args.deepmath_samples,
        "math": args.math_samples,
        "numinamath": args.numinamath_samples,
    }

    # Validate ratio: MATH samples × 2 should match DeepMath and NuminaMath.
    expected_deepmath = args.math_samples * 2
    expected_numinamath = args.math_samples * 2
    assert args.deepmath_samples == expected_deepmath, (
        f"1:2:2 ratio violation: --deepmath-samples ({args.deepmath_samples}) "
        f"must equal 2 × --math-samples ({expected_deepmath})"
    )
    assert args.numinamath_samples == expected_numinamath, (
        f"1:2:2 ratio violation: --numinamath-samples ({args.numinamath_samples}) "
        f"must equal 2 × --math-samples ({expected_numinamath})"
    )

    rows_by_dataset: dict[str, list[dict[str, object]]] = {}
    for dataset_name in IN_DISTRIBUTION_DATASET_ORDER:
        dataset = datasets[dataset_name]
        count = sample_counts[dataset_name]
        assert_enough_rows(dataset_name, dataset.num_rows, 0, count)
        rows_by_dataset[dataset_name] = build_dataset_entries(
            dataset_name,
            dataset,
            start_question_id=0,
            target_count=count,
        )

    ordered_rows = interleave_rows_weighted(
        rows_by_dataset, pattern=TRAIN_INTERLEAVE_PATTERN
    )
    write_jsonl_entries(output_path, ordered_rows)
    print(f"Wrote {len(ordered_rows)} rows to {output_path.resolve()}")
    for name in IN_DISTRIBUTION_DATASET_ORDER:
        print(f"  {name}: {sample_counts[name]} samples")


if __name__ == "__main__":
    main()
