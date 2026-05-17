"""Generate a GSM8K test sqlite file using TestSetEntry payloads."""

from __future__ import annotations

from pathlib import Path

from test_set_sqlite_common import (
    build_and_write_test_sqlite,
    create_parser,
    install_hf_token,
    load_gsm8k_test,
)


def main() -> None:
    parser = create_parser("gsm8k")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    install_hf_token(repo_root)

    dataset = load_gsm8k_test()
    output_path = args.output or (repo_root / "datasets" / "gsm8k_test.sqlite")
    source_rows, written_rows, repeated = build_and_write_test_sqlite(
        "gsm8k",
        dataset,
        output_path,
        source_split="test",
        sample_seed=args.sample_seed,
    )

    print(f"Wrote {written_rows} rows to {output_path.resolve()}")
    if repeated:
        print(
            f"Note: source split has {source_rows} rows; repeated from start to reach {written_rows} rows."
        )


if __name__ == "__main__":
    main()
