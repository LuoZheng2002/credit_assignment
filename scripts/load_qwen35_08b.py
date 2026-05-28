#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download Qwen3.5-0.8B from Hugging Face and save it locally in "
            "Transformers format using safetensors weights."
        )
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        required=True,
        help=(
            "Parent directory"
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        help=f"Model to download",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to from_pretrained calls",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # output_path = args.output_path / f"model_{args.model.split('/')[-1].lower().replace('.', '_')}"
    output_path = args.output_path / "model"
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
    )

    print(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
    )

    print(f"Saving tokenizer to: {output_path}")
    tokenizer.save_pretrained(output_path)

    print(f"Saving model to: {output_path}")
    model.save_pretrained(output_path, safe_serialization=True)

    print(f"Done. Local model folder is ready at: {output_path}")


if __name__ == "__main__":
    main()
