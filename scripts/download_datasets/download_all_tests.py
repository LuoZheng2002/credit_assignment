"""Generate all test sqlite files in one command."""

from __future__ import annotations

import argparse
from pathlib import Path

from test_set_sqlite_common import (
    DEFAULT_SAMPLE_SEED,
    build_and_write_test_sqlite,
    install_hf_token,
    load_aime24_test,
    load_aime25_test,
    load_amc23_test,
    load_deepmath_test,
    load_gsm8k_test,
    load_math_test,
)


def _run_one(repo_root: Path, dataset_name: str, dataset: object, source_split: str, sample_seed: int) -> None:
    output_path = repo_root / "datasets" / f"{dataset_name}_test.sqlite"
    source_rows, written_rows, repeated = build_and_write_test_sqlite(
        dataset_name,
        dataset,
        output_path,
        source_split=source_split,
        sample_seed=sample_seed,
    )
    print(f"[{dataset_name}] Wrote {written_rows} rows to {output_path.resolve()}")
    if repeated:
        print(
            f"[{dataset_name}] Note: source split has {source_rows} rows; "
            f"repeated from start to reach {written_rows} rows."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and build all *_test.sqlite files")
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=DEFAULT_SAMPLE_SEED,
        help=f"Random seed used when sampling from large test splits (default: {DEFAULT_SAMPLE_SEED})",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    install_hf_token(repo_root)

    deepmath, deepmath_source_split = load_deepmath_test()
    _run_one(repo_root, "deepmath", deepmath, deepmath_source_split, args.sample_seed)
    if deepmath_source_split != "test":
        print("[deepmath] Note: DeepMath has no test split; train split was used as the source.")

    math = load_math_test()
    _run_one(repo_root, "math", math, "test", args.sample_seed)

    gsm8k = load_gsm8k_test()
    _run_one(repo_root, "gsm8k", gsm8k, "test", args.sample_seed)

    aime24 = load_aime24_test()
    _run_one(repo_root, "aime24", aime24, "test", args.sample_seed)

    aime25 = load_aime25_test()
    _run_one(repo_root, "aime25", aime25, "test", args.sample_seed)

    amc23 = load_amc23_test()
    _run_one(repo_root, "amc23", amc23, "test", args.sample_seed)


if __name__ == "__main__":
    main()
