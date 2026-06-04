"""Shared utilities for generating per-dataset test-set sqlite files."""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

from datasets import concatenate_datasets, load_dataset
from dotenv import load_dotenv
from research_utility import SqliteStore

MIN_SAMPLES = 1_000
DEFAULT_SAMPLE_SEED = 42
TRAIN_SPLIT_CLEARANCE = 5_000
TRAINING_DATASET_NAMES = {"deepmath", "math", "gsm8k"}


def install_hf_token(repo_root: Path) -> None:
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


def parse_gsm8k_final_answer(answer: str) -> str:
    parts = answer.split("####", 1)
    if len(parts) == 2:
        return parts[1].strip()
    return answer.strip()


def get_required_field(raw: dict[str, object], dataset_name: str, field_candidates: list[str]) -> str:
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


def load_deepmath_test() -> tuple[object, str]:
    try:
        return load_dataset("mlfoundations-dev/deepmath", split="test"), "test"
    except Exception:
        return load_dataset("mlfoundations-dev/deepmath", split="train"), "train"


def load_math_test() -> object:
    try:
        return load_dataset("EleutherAI/hendrycks_math", "all", split="test")
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
        parts = [load_dataset("EleutherAI/hendrycks_math", cfg, split="test") for cfg in configs]
        return concatenate_datasets(parts)


def load_gsm8k_test() -> object:
    main = load_dataset("openai/gsm8k", "main", split="test")
    socratic = load_dataset("openai/gsm8k", "socratic", split="test")
    return concatenate_datasets([main, socratic])


def load_aime24_test() -> object:
    return load_dataset("math-ai/aime24", "default", split="test")


def load_aime25_test() -> object:
    return load_dataset("math-ai/aime25", "default", split="test")


def load_amc23_test() -> object:
    return load_dataset("math-ai/amc23", "default", split="test")


def normalize_entry(dataset_name: str, raw: dict[str, object]) -> tuple[str, str]:
    if dataset_name == "deepmath":
        question = get_required_field(raw, dataset_name, ["question"])
        answer = get_required_field(raw, dataset_name, ["final_answer", "answer"])
        return question, answer

    if dataset_name == "math":
        question = get_required_field(raw, dataset_name, ["problem"])
        solution = get_required_field(raw, dataset_name, ["solution"])
        return question, extract_boxed_content(solution) or solution.strip()

    if dataset_name == "gsm8k":
        question = get_required_field(raw, dataset_name, ["question"])
        answer = get_required_field(raw, dataset_name, ["answer"])
        return question, parse_gsm8k_final_answer(answer)

    if dataset_name == "aime24":
        question = get_required_field(raw, dataset_name, ["problem", "question"])
        solution = get_required_field(raw, dataset_name, ["solution"])
        return question, extract_boxed_content(solution) or solution.strip()

    if dataset_name == "aime25":
        question = get_required_field(raw, dataset_name, ["problem", "question"])
        answer = get_required_field(raw, dataset_name, ["answer", "final_answer"])
        return question, answer

    if dataset_name == "amc23":
        question = get_required_field(raw, dataset_name, ["question", "problem"])
        answer = get_required_field(raw, dataset_name, ["answer", "final_answer"])
        return question, answer

    raise AssertionError(f"Unknown dataset_name: {dataset_name}")


def create_parser(dataset_name: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"Download and build {dataset_name}_test.sqlite")
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=DEFAULT_SAMPLE_SEED,
        help=f"Random seed used when sampling from large test splits (default: {DEFAULT_SAMPLE_SEED})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=f"Output sqlite path (default: <repo>/datasets/{dataset_name}_test.sqlite)",
    )
    return parser


def write_store_entries(db_path: Path, payload_rows: list[dict[str, object]]) -> None:
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


def _sample_question_ids(
    dataset_name: str,
    num_rows: int,
    target_count: int,
    source_split: str,
    sample_seed: int,
    clearance_override: int | None = None,
) -> list[int]:
    assert num_rows > 0, f"{dataset_name} dataset must contain at least one row"

    clearance_start = 0
    if source_split == "train" and dataset_name in TRAINING_DATASET_NAMES:
        effective_clearance = (
            clearance_override if clearance_override is not None else TRAIN_SPLIT_CLEARANCE
        )
        clearance_start = min(effective_clearance, num_rows)

    if num_rows > MIN_SAMPLES:
        candidate_ids = list(range(clearance_start, num_rows))
        assert candidate_ids, (
            f"No rows left after reserving the first {TRAIN_SPLIT_CLEARANCE} train rows for {dataset_name}"
        )

        rng = random.Random(sample_seed)
        if target_count <= len(candidate_ids):
            sampled = rng.sample(candidate_ids, target_count)
        else:
            sampled = candidate_ids.copy()
            sampled.extend(rng.choices(candidate_ids, k=target_count - len(candidate_ids)))

        sampled.sort()
        return sampled

    base_start = clearance_start if clearance_start < num_rows else 0
    width = num_rows - base_start
    assert width > 0, f"No usable rows found for {dataset_name}"
    return [base_start + (i % width) for i in range(target_count)]


def build_and_write_test_sqlite(
    dataset_name: str,
    dataset: object,
    output_path: Path,
    source_split: str,
    sample_seed: int,
    clearance_override: int | None = None,
) -> tuple[int, int, bool]:
    num_rows = dataset.num_rows
    assert num_rows > 0, f"{dataset_name} dataset must contain at least one row"

    target_count = min(num_rows, MIN_SAMPLES)
    repeated = target_count > num_rows
    question_ids = _sample_question_ids(dataset_name, num_rows, target_count, source_split, sample_seed, clearance_override=clearance_override)

    rows: list[dict[str, object]] = []
    for flat_id, question_id in enumerate(question_ids):
        raw = dataset[question_id]
        question, correct_answer = normalize_entry(dataset_name, raw)
        rows.append(
            {
                "flat_id": flat_id,
                "question_id": question_id,
                "question": question,
                "correct_answer": correct_answer,
            }
        )

    write_store_entries(output_path, rows)
    return num_rows, target_count, repeated
