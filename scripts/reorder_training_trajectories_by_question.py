#!/usr/bin/env python3
"""Reorder a generated training trajectory msgpack by question flat id.

This is useful when the original action log has been pruned but an existing
trajectory file can be reused for a new sample-ordering experiment.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import msgpack


def _question_flat_id(record: Any) -> int:
    if not isinstance(record, dict):
        raise TypeError(f"expected msgpack record dict, got {type(record).__name__}")
    question = record.get("question")
    if not isinstance(question, dict):
        raise KeyError("record does not contain a question object")
    flat_id = question.get("flat_id")
    if isinstance(flat_id, int):
        return flat_id
    if isinstance(flat_id, dict) and "0" in flat_id:
        return int(flat_id["0"])
    return int(flat_id)


def _trajectory_length(record: Any) -> int:
    if not isinstance(record, dict):
        return 0
    input_ids = record.get("input_ids")
    if hasattr(input_ids, "__len__"):
        return len(input_ids)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    source_msgpack = args.source_dir / "training_trajectories" / "trajectories.msgpack"
    source_stats = args.source_dir / "training_trajectories_stats.json"
    source_bundle = args.source_dir / "training_trajectories" / "config_bundle.json"
    output_trajectories_dir = args.output_dir / "training_trajectories"
    output_msgpack = output_trajectories_dir / "trajectories.msgpack"
    output_stats = args.output_dir / "training_trajectories_stats.json"
    output_bundle = output_trajectories_dir / "config_bundle.json"
    output_metadata = args.output_dir / "reorder_metadata.json"

    if not source_msgpack.exists():
        raise FileNotFoundError(source_msgpack)

    output_trajectories_dir.mkdir(parents=True, exist_ok=True)
    records: list[tuple[int, int, int, Any]] = []
    with source_msgpack.open("rb") as handle:
        unpacker = msgpack.Unpacker(handle, raw=False, strict_map_key=False)
        for index, record in enumerate(unpacker):
            records.append((_question_flat_id(record), index, _trajectory_length(record), record))

    records.sort(key=lambda item: (item[0], item[1]))
    with output_msgpack.open("wb") as handle:
        packer = msgpack.Packer(use_bin_type=True)
        for _, _, _, record in records:
            handle.write(packer.pack(record))

    if source_stats.exists():
        shutil.copy2(source_stats, output_stats)
    if source_bundle.exists():
        shutil.copy2(source_bundle, output_bundle)

    lengths = [length for _, _, length, _ in records]
    metadata = {
        "source_msgpack": str(source_msgpack),
        "output_msgpack": str(output_msgpack),
        "sort_mode": "ByQuestionFromGeneratedTrajectories",
        "num_records": len(records),
        "num_questions": len({question_id for question_id, _, _, _ in records}),
        "first_lengths": lengths[:20],
        "last_lengths": lengths[-20:],
    }
    output_metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
