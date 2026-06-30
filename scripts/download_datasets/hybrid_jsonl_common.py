"""Shared utilities for building hybrid dataset JSONL files.

This module centralizes dataset loading, schema normalization, sampling, and
JSONL writing for the hybrid train/val/test generation scripts.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any


def _normalize_proxy_env() -> None:
    for key in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ):
        value = os.environ.get(key)
        if value and value.startswith("socks://"):
            os.environ[key] = "socks5://" + value[len("socks://") :]


# Set before importing `datasets` so that huggingface_hub picks it up.
_normalize_proxy_env()
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from dotenv import load_dotenv

from datasets import concatenate_datasets, load_dataset

IN_DISTRIBUTION_DATASET_ORDER = ("deepmath", "math", "numinamath")
EVALUATION_DATASET_ORDER = (
    "deepmath",
    "math",
    "numinamath",
    "amc2023",
    "gaokao_math_2024",
    "collegemath",
)
IN_DISTRIBUTION_DATASET_NAMES = set(IN_DISTRIBUTION_DATASET_ORDER)

DEFAULT_TRAIN_SAMPLES_PER_DATASET = 5_000
DEFAULT_VAL_SAMPLES_PER_DATASET = 1_000
DEFAULT_TEST_SAMPLES_PER_DATASET = 1_000
DEFAULT_SAMPLE_SEED = 42


MATH_CONFIGS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]


def install_hf_token(repo_root: Path) -> None:
    print(f"HF_ENDPOINT={os.environ['HF_ENDPOINT']}")
    env_file = repo_root / ".env"
    assert env_file.exists(), "Missing .env file; cannot load HF_TOKEN"
    load_dotenv(env_file, override=True)
    assert os.environ.get("HF_TOKEN"), "HF_TOKEN not defined in .env"


def extract_boxed_content(text: str) -> str | None:
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


def get_required_field(
    raw: dict[str, Any], dataset_name: str, field_candidates: list[str]
) -> str:
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


def normalize_numeric_answer(value: Any) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def load_math_split(split: str) -> Any:
    try:
        return load_dataset("EleutherAI/hendrycks_math", "all", split=split)
    except Exception:
        parts = [
            load_dataset("EleutherAI/hendrycks_math", cfg, split=split)
            for cfg in MATH_CONFIGS
        ]
        return concatenate_datasets(parts)


def load_deepmath_train() -> Any:
    return load_dataset("mlfoundations-dev/deepmath", split="train")


def load_deepmath_test() -> tuple[Any, str]:
    try:
        return load_dataset("mlfoundations-dev/deepmath", split="test"), "test"
    except Exception:
        return load_deepmath_train(), "train"


def load_numinamath_train() -> Any:
    return load_dataset("AI-MO/NuminaMath-CoT", split="train")


def load_numinamath_test() -> Any:
    return load_dataset("AI-MO/NuminaMath-CoT", split="test")


def load_amc2023_test() -> Any:
    return load_dataset("sparkle-reasoning/amc2023", split="test")


def load_gaokao_math_2024_test() -> Any:
    dataset = load_dataset("FrankieYao/GaoKaoMath", split="test")
    if "year" in dataset.column_names:
        dataset = dataset.filter(lambda row: str(row["year"]) == "2024")
    assert dataset.num_rows > 0, "GaoKaoMath 2024 test subset must be non-empty"
    return dataset


def load_collegemath_test() -> Any:
    return load_dataset("realtreetune/college_math", split="test")


def load_in_distribution_datasets() -> dict[str, Any]:
    return {
        "deepmath": load_deepmath_train(),
        "math": load_math_split("train"),
        "numinamath": load_numinamath_train(),
    }


def load_evaluation_dataset(dataset_name: str) -> tuple[Any, str]:
    if dataset_name == "deepmath":
        return load_deepmath_test()
    if dataset_name == "math":
        return load_math_split("test"), "test"
    if dataset_name == "numinamath":
        return load_numinamath_test(), "test"
    if dataset_name == "amc2023":
        return load_amc2023_test(), "test"
    if dataset_name == "gaokao_math_2024":
        return load_gaokao_math_2024_test(), "test"
    if dataset_name == "collegemath":
        return load_collegemath_test(), "test"
    raise AssertionError(f"Unknown dataset_name: {dataset_name}")


def normalize_entry(dataset_name: str, raw: dict[str, Any]) -> tuple[str, str]:
    if dataset_name == "deepmath":
        question = get_required_field(raw, dataset_name, ["question"])
        answer = get_required_field(raw, dataset_name, ["final_answer", "answer"])
        return question, answer.strip()

    if dataset_name == "math":
        question = get_required_field(raw, dataset_name, ["problem"])
        solution = get_required_field(raw, dataset_name, ["solution"])
        return question, extract_boxed_content(solution) or solution.strip()

    if dataset_name == "numinamath":
        question = get_required_field(raw, dataset_name, ["problem"])
        solution = get_required_field(raw, dataset_name, ["solution"])
        return question, extract_boxed_content(solution) or solution.strip()

    if dataset_name == "amc2023":
        question = get_required_field(raw, dataset_name, ["question", "problem"])
        answer = raw.get("answer")
        assert answer is not None, "amc2023 entry missing answer"
        return question, normalize_numeric_answer(answer)

    if dataset_name == "gaokao_math_2024":
        question = get_required_field(raw, dataset_name, ["problem", "question"])
        answer = get_required_field(raw, dataset_name, ["answer"])
        return question, answer.strip()

    if dataset_name == "collegemath":
        question = get_required_field(raw, dataset_name, ["question", "problem"])
        answer = get_required_field(raw, dataset_name, ["answer"])
        return question, answer.strip()

    raise AssertionError(f"Unknown dataset_name: {dataset_name}")


def assert_enough_rows(
    dataset_name: str, num_rows: int, reserved_count: int, requested_count: int
) -> None:
    assert reserved_count >= 0, "reserved_count must be non-negative"
    assert requested_count > 0, "requested_count must be positive"
    required = reserved_count + requested_count
    assert num_rows >= required, (
        f"{dataset_name} split has {num_rows} rows, but requires at least {required} "
        f"({reserved_count} reserved + {requested_count} requested) to avoid overlap"
    )


def build_dataset_entries(
    dataset_name: str,
    dataset: Any,
    start_question_id: int,
    target_count: int,
) -> list[dict[str, Any]]:
    assert target_count > 0, "target_count must be positive"
    num_rows = dataset.num_rows
    assert start_question_id >= 0, "start_question_id must be non-negative"
    assert start_question_id + target_count <= num_rows, (
        f"{dataset_name} does not have enough rows for requested slice: "
        f"start={start_question_id}, count={target_count}, rows={num_rows}"
    )

    rows: list[dict[str, Any]] = []
    for question_id in range(start_question_id, start_question_id + target_count):
        raw = dataset[question_id]
        question, correct_answer = normalize_entry(dataset_name, raw)
        rows.append(
            {
                "dataset_name": dataset_name,
                "question_id": question_id,
                "question": question,
                "correct_answer": correct_answer,
            }
        )
    return rows


def write_jsonl_entries(output_path: Path, payload_rows: list[dict[str, Any]]) -> None:
    if output_path.exists():
        output_path.unlink()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for row in payload_rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            f.write("\n")


def interleave_rows(
    rows_by_dataset: dict[str, list[dict[str, Any]]],
    dataset_order: tuple[str, ...],
    rows_per_dataset: int,
) -> list[dict[str, Any]]:
    ordered_rows: list[dict[str, Any]] = []
    for i in range(rows_per_dataset):
        for dataset_name in dataset_order:
            source = rows_by_dataset[dataset_name][i]
            ordered_rows.append(
                {
                    "flat_id": len(ordered_rows),
                    "dataset_name": source["dataset_name"],
                    "question_id": source["question_id"],
                    "question": source["question"],
                    "correct_answer": source["correct_answer"],
                }
            )
    return ordered_rows


def sample_question_ids(
    *,
    start_question_id: int,
    num_rows: int,
    target_count: int,
    sample_seed: int,
) -> list[int]:
    assert num_rows > 0, "dataset must contain at least one row"
    assert start_question_id >= 0, "start_question_id must be non-negative"
    assert start_question_id < num_rows, (
        "start_question_id must be within dataset bounds"
    )
    assert target_count > 0, "target_count must be positive"

    candidate_ids = list(range(start_question_id, num_rows))
    assert candidate_ids, "No candidate rows available for sampling"
    assert target_count <= len(candidate_ids), (
        f"Requested {target_count} rows, but only {len(candidate_ids)} non-overlapping rows are available"
    )

    if target_count == len(candidate_ids):
        return candidate_ids

    rng = random.Random(sample_seed)
    sampled = rng.sample(candidate_ids, target_count)
    sampled.sort()
    return sampled
