#!/usr/bin/env python3
import argparse
import sys
from typing import List

try:
    from transformers import AutoTokenizer
except ModuleNotFoundError as exc:
    print(
        "Missing dependency: transformers. Install it first, for example:\n"
        "  pip install transformers sentencepiece\n"
        "Then rerun this script.",
        file=sys.stderr,
    )
    raise SystemExit(1) from exc


DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"


def parse_token_ids(args_ids: List[str], csv_ids: str | None) -> List[int]:
    ids: List[int] = []
    if csv_ids:
        ids.extend(int(chunk.strip()) for chunk in csv_ids.split(",") if chunk.strip())
    ids.extend(int(x) for x in args_ids)
    if not ids:
        ids = [624, 11135, 697, 32711, 25]
    return ids


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Decode Qwen2.5 token IDs and inspect per-token text."
    )
    parser.add_argument("token_ids", nargs="*", help="Token IDs as positional ints")
    parser.add_argument(
        "--ids",
        type=str,
        default=None,
        help="Comma-separated token IDs, e.g. '624,11135,697,32711,25'",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Tokenizer model name (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to AutoTokenizer.from_pretrained",
    )
    args = parser.parse_args()

    token_ids = parse_token_ids(args.token_ids, args.ids)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
    )

    decoded = tokenizer.decode(
        token_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )

    print(f"model: {args.model}")
    print(f"token_ids: {token_ids}")
    print(f"decoded repr: {decoded!r}")
    print("decoded text:")
    print(decoded)
    print("per-token decode:")

    for idx, token_id in enumerate(token_ids):
        piece = tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        print(f"  {idx:>3}: id={token_id:<8} piece={piece!r}")


if __name__ == "__main__":
    main()
