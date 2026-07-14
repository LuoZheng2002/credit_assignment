"""Unified tokenizer downloader.

Usage:
    python scripts/download_tokenizer.py --model <hf_model_id>

Downloads the tokenizer from Hugging Face and saves it under `tokenizers/<dir>/`
along with the chat template.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make `scripts/` importable so we can pull in `_bootstrap`.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: F401

from transformers import AutoTokenizer  # noqa: E402

# Mapping from HuggingFace model ID → local subdirectory under `tokenizers/`.
# Variants that share the same tokenizer (e.g. Qwen3-4B / Qwen3-0.6B) map to
# the same directory.
MODEL_TO_DIR: dict[str, str] = {
    "google/gemma-3-4b-it": "gemma3",
    "meta-llama/Llama-3.1-8B-Instruct": "llama31",
    "mistralai/Mistral-7B-Instruct-v0.3": "mistral7b",
    "Qwen/Qwen2.5-7B-Instruct": "qwen25",
    "Qwen/Qwen3-4B": "qwen3",
    "Qwen/Qwen3-0.6B": "qwen3",
    "Qwen/Qwen3.5-4B": "qwen35",
    "Qwen/Qwen3.5-0.8B": "qwen35",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download a tokenizer from Hugging Face"
    )
    parser.add_argument(
        "--model", required=True, help="Hugging Face model ID (e.g. google/gemma-3-4b-it)"
    )
    args = parser.parse_args()

    model_id: str = args.model
    dir_name = MODEL_TO_DIR.get(model_id)
    if dir_name is None:
        print(
            f"Error: unknown model '{model_id}'. "
            f"Known models: {', '.join(sorted(MODEL_TO_DIR))}",
            file=sys.stderr,
        )
        sys.exit(1)

    output_dir = Path("tokenizers") / dir_name

    tokenizer_json = output_dir / "tokenizer.json"
    chat_template_jinja = output_dir / "chat_template.jinja"
    if tokenizer_json.exists() and chat_template_jinja.exists():
        print(f"Tokenizer for {model_id} already exists at {output_dir}")
        return

    print(f"Downloading tokenizer for {model_id} …")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(str(output_dir))

    chat_template = tokenizer.chat_template
    if chat_template is None:
        raise RuntimeError(f"Tokenizer chat_template is missing for {model_id}")

    chat_template_jinja.write_text(chat_template, encoding="utf-8")
    print(f"Saved tokenizer to {output_dir}")


if __name__ == "__main__":
    main()
