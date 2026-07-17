#!/usr/bin/env python3
"""Convert this repo's hybrid JSONL datasets to VERL-compatible parquet.

This intentionally bypasses the existing rollout/generation/training pipeline.
It only reuses the raw hybrid_{train,val}.jsonl files and writes the schema
expected by VERL's RLHF dataset loader.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import Dataset

INSTRUCTION = (
    "Let's solve the problem step by step. Put the final answer after exactly "
    'four hash marks, like "#### final answer".'
)


def _read_jsonl(path: Path, *, limit: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_index, line in enumerate(handle):
            if limit is not None and len(rows) >= limit:
                break
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on {path}:{line_index + 1}: {exc}") from exc
    return rows


def _convert_row(row: dict[str, Any], *, split: str, index: int) -> dict[str, Any]:
    dataset_name = str(row["dataset_name"])
    question = str(row["question"])
    correct_answer = str(row["correct_answer"])
    content = f"{question}\n\n{INSTRUCTION}"
    return {
        "data_source": dataset_name,
        "prompt": [{"role": "user", "content": content}],
        "ability": "math",
        "reward_model": {
            "style": "rule",
            "ground_truth": correct_answer,
        },
        "extra_info": {
            "split": split,
            "index": index,
            "flat_id": row.get("flat_id"),
            "question_id": row.get("question_id"),
            "dataset_name": dataset_name,
        },
    }


def _write_split(input_path: Path, output_path: Path, *, split: str, limit: int | None) -> None:
    rows = _read_jsonl(input_path, limit=limit)
    converted = [_convert_row(row, split=split, index=index) for index, row in enumerate(rows)]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(converted).to_parquet(str(output_path))
    print(f"wrote {len(converted)} {split} rows to {output_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-jsonl", type=Path, default=Path("datasets/hybrid_train.jsonl"))
    parser.add_argument("--val-jsonl", type=Path, default=Path("datasets/hybrid_val.jsonl"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--val-limit", type=int, default=None)
    args = parser.parse_args()

    _write_split(
        args.train_jsonl,
        args.output_dir / "train.parquet",
        split="train",
        limit=args.train_limit,
    )
    _write_split(
        args.val_jsonl,
        args.output_dir / "val.parquet",
        split="val",
        limit=args.val_limit,
    )


if __name__ == "__main__":
    main()
