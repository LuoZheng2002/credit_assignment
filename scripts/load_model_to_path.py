#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download a model snapshot from Hugging Face to a local folder."
        )
    )
    parser.add_argument(
        "--output-parent-dir",
        type=Path,
        required=True,
        help=(
            "Parent directory where a 'model' subfolder will be created"
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        help=f"Model to download",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if os.environ.get("HF_TOKEN") in {None, ""}:
        print(
            "Warning: HF_TOKEN is not set. Authenticated Hugging Face downloads are typically faster and less rate-limited."
        )

    output_path = args.output_parent_dir / "model"
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Downloading model snapshot: {args.model}")
    snapshot_download(
        repo_id=args.model,
        local_dir=output_path,
        token=os.environ.get("HF_TOKEN") or None,
    )

    print(f"Done. Local model folder is ready at: {output_path}")


if __name__ == "__main__":
    main()
