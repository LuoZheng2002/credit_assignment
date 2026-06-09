from __future__ import annotations

import argparse
import os
from pathlib import Path


def _has_hf_model_weights(model_dir: Path) -> bool:
    return (model_dir / "model.safetensors").is_file() or (
        model_dir / "model.safetensors.index.json"
    ).is_file()


def _remove_redundant_consolidated_weights(model_dir: Path) -> int:
    if not _has_hf_model_weights(model_dir):
        return 0

    removed = 0
    for pattern in ("consolidated*.safetensors", "consolidated*.safetensors.index.json"):
        for file_path in model_dir.glob(pattern):
            if file_path.is_file():
                file_path.unlink()
                removed += 1
    return removed


def ensure_model_snapshot(output_parent_dir: Path, model_name: str) -> Path:
    from huggingface_hub import snapshot_download

    output_path = output_parent_dir / "model"
    output_path.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=model_name,
        local_dir=output_path,
        token=os.environ.get("HF_TOKEN") or None,
    )
    _remove_redundant_consolidated_weights(output_path)
    return output_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download a model snapshot from Hugging Face to a local folder"
    )
    parser.add_argument(
        "--output-parent-dir",
        type=Path,
        required=True,
        help="Parent directory where a 'model' subfolder will be created",
    )
    parser.add_argument("--model", type=str, required=True, help="Model repo id to download")
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    if os.environ.get("HF_TOKEN") in {None, ""}:
        print(
            "Warning: HF_TOKEN is not set. Authenticated Hugging Face downloads are typically faster and less rate-limited."
        )

    print(f"Downloading model snapshot: {args.model}")
    output_path = ensure_model_snapshot(args.output_parent_dir, args.model)
    print(f"Done. Local model folder is ready at: {output_path}")


if __name__ == "__main__":
    main()
