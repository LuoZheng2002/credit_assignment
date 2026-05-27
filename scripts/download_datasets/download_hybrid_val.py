"""Build a hybrid validation sqlite database from DeepMath, MATH, and GSM8K."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from datasets import concatenate_datasets, load_dataset
from dotenv import load_dotenv
from research_utility import SqliteStore

TRAIN_SAMPLES_PER_DATASET = 5_000
VAL_SAMPLES_PER_DATASET = 1_000


def _install_hf_token(repo_root: Path) -> None:
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


def _load_math_train() -> object:
    try:
        return load_dataset("EleutherAI/hendrycks_math", "all", split="train")
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
        return concatenate_datasets(parts)


def _assert_enough_rows(dataset_name: str, num_rows: int, train_count: int, val_count: int) -> None:
    required = train_count + val_count
    assert num_rows >= required, (
        f"{dataset_name} train split has {num_rows} rows, but requires at least {required} "
        f"({train_count} train + {val_count} val) to avoid overlap"
    )


def _build_dataset_entries(
    dataset_name: str,
    dataset: object,
    start_question_id: int,
    target_count: int,
) -> list[dict[str, object]]:
    assert target_count > 0, "target_count must be positive"
    num_rows = dataset.num_rows
    assert start_question_id >= 0, "start_question_id must be non-negative"
    assert start_question_id + target_count <= num_rows, (
        f"{dataset_name} does not have enough rows for requested slice: "
        f"start={start_question_id}, count={target_count}, rows={num_rows}"
    )

    rows: list[dict[str, object]] = []
    for i in range(target_count):
        question_id = start_question_id + i
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

    store = SqliteStore[str, dict[str, object]](db_path)
    try:
        for row in payload_rows:
            flat_id = row["flat_id"]
            store.upsert(str(flat_id), row)
    finally:
        store.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and build hybrid_val.sqlite")
    parser.add_argument(
        "--samples-per-dataset",
        type=int,
        default=VAL_SAMPLES_PER_DATASET,
        help=f"Number of validation samples per dataset (default: {VAL_SAMPLES_PER_DATASET})",
    )
    parser.add_argument(
        "--train-samples-per-dataset",
        type=int,
        default=TRAIN_SAMPLES_PER_DATASET,
        help=(
            "Number of train samples already reserved per dataset "
            f"(default: {TRAIN_SAMPLES_PER_DATASET})"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output sqlite path (default: <repo>/datasets/hybrid_val.sqlite)",
    )
    args = parser.parse_args()

    assert args.samples_per_dataset > 0, "--samples-per-dataset must be positive"
    assert args.train_samples_per_dataset >= 0, "--train-samples-per-dataset must be non-negative"

    repo_root = Path(__file__).resolve().parents[2]
    _install_hf_token(repo_root)

    output_path = args.output or (repo_root / "datasets" / "hybrid_val.sqlite")

    deepmath = load_dataset("mlfoundations-dev/deepmath", split="train")
    math = _load_math_train()
    gsm8k = load_dataset("openai/gsm8k", "main", split="train")

    _assert_enough_rows("deepmath", deepmath.num_rows, args.train_samples_per_dataset, args.samples_per_dataset)
    _assert_enough_rows("math", math.num_rows, args.train_samples_per_dataset, args.samples_per_dataset)
    _assert_enough_rows("gsm8k", gsm8k.num_rows, args.train_samples_per_dataset, args.samples_per_dataset)

    deepmath_rows = _build_dataset_entries(
        "deepmath",
        deepmath,
        start_question_id=args.train_samples_per_dataset,
        target_count=args.samples_per_dataset,
    )
    math_rows = _build_dataset_entries(
        "math",
        math,
        start_question_id=args.train_samples_per_dataset,
        target_count=args.samples_per_dataset,
    )
    gsm8k_rows = _build_dataset_entries(
        "gsm8k",
        gsm8k,
        start_question_id=args.train_samples_per_dataset,
        target_count=args.samples_per_dataset,
    )

    ordered_rows: list[dict[str, object]] = []
    interleave_order = ("deepmath", "math", "gsm8k")

    for i in range(args.samples_per_dataset):
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


if __name__ == "__main__":
    main()
