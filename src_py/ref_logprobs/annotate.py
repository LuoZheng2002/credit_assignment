from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable

import msgpack
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

IGNORE_LABEL = -100
CHUNK_RE = re.compile(r"^chunk_(\d+)\.msgpack$")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Annotate generated training trajectories with offline reference logprobs."
    )
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--model-path-or-name", required=True)
    parser.add_argument("--fallback-model-name", required=True)
    parser.add_argument("--max-batch-tokens", type=int, default=8192)
    parser.add_argument("--login-smoke", action="store_true")
    return parser.parse_args()


def _iter_msgpack(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("rb") as file:
        unpacker = msgpack.Unpacker(file, raw=False)
        for payload in unpacker:
            assert isinstance(payload, dict), f"trajectory payload must be dict: {path}"
            yield payload


def _write_msgpack(path: Path, payloads: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with tmp_path.open("wb") as file:
        packer = msgpack.Packer(use_bin_type=True)
        for payload in payloads:
            file.write(packer.pack(payload))
            count += 1
    tmp_path.replace(path)
    return count


def _trajectory_files(input_dir: Path) -> list[tuple[Path, Path]]:
    files: list[tuple[Path, Path]] = []
    global_path = input_dir / "trajectories.msgpack"
    if global_path.exists():
        files.append((global_path, input_dir / "trajectories_ref_logprobs.msgpack"))

    chunk_paths: list[tuple[int, Path]] = []
    for path in input_dir.glob("chunk_*.msgpack"):
        match = CHUNK_RE.match(path.name)
        if match is not None:
            chunk_paths.append((int(match.group(1)), path))
    for chunk_index, path in sorted(chunk_paths):
        files.append((path, input_dir / f"chunk_{chunk_index}_ref_logprobs.msgpack"))
    return files


def _load_model(model_path_or_name: str, fallback_model_name: str) -> tuple[Any, Any]:
    model_source = model_path_or_name if Path(model_path_or_name).exists() else fallback_model_name
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_source)
    except Exception as exc:
        if model_source == fallback_model_name:
            raise
        print(
            "local_tokenizer_load_failed=1 "
            f"model_source={model_source} fallback_model_name={fallback_model_name} "
            f"error_type={type(exc).__name__} error_message={exc}",
            flush=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(fallback_model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return tokenizer, model


def _payload_length(payload: dict[str, Any]) -> int:
    input_ids = payload.get("input_ids")
    assert isinstance(input_ids, list), "payload input_ids must be list"
    return len(input_ids)


def _iter_batches(payloads: list[dict[str, Any]], max_batch_tokens: int) -> Iterable[list[dict[str, Any]]]:
    assert max_batch_tokens > 0, "max_batch_tokens must be positive"
    batch: list[dict[str, Any]] = []
    max_len = 0
    for payload in payloads:
        length = _payload_length(payload)
        candidate_max_len = max(max_len, length)
        if batch and candidate_max_len * (len(batch) + 1) > max_batch_tokens:
            yield batch
            batch = []
            max_len = 0
        batch.append(payload)
        max_len = max(max_len, length)
    if batch:
        yield batch


def _annotate_batch(batch: list[dict[str, Any]], model: Any, pad_token_id: int) -> list[dict[str, Any]]:
    max_len = max(_payload_length(payload) for payload in batch)
    input_rows: list[list[int]] = []
    attention_rows: list[list[int]] = []
    label_rows: list[list[int]] = []
    for payload in batch:
        input_ids = payload["input_ids"]
        labels = payload["labels"]
        assert isinstance(input_ids, list), "input_ids must be list"
        assert isinstance(labels, list), "labels must be list"
        assert len(input_ids) == len(labels), "input_ids and labels must align"
        pad_count = max_len - len(input_ids)
        input_rows.append(input_ids + [pad_token_id] * pad_count)
        label_rows.append(labels + [IGNORE_LABEL] * pad_count)
        attention_rows.append([1] * len(input_ids) + [0] * pad_count)

    device = next(model.parameters()).device
    input_tensor = torch.tensor(input_rows, dtype=torch.long, device=device)
    label_tensor = torch.tensor(label_rows, dtype=torch.long, device=device)
    attention_tensor = torch.tensor(attention_rows, dtype=torch.long, device=device)

    with torch.inference_mode():
        outputs = model(input_ids=input_tensor, attention_mask=attention_tensor, use_cache=False)
        logits = outputs.logits
        shifted_logits = logits[:, :-1, :]
        shifted_input_ids = input_tensor[:, 1:]
        shifted_labels = label_tensor[:, 1:]
        supervised_mask = shifted_labels.ne(IGNORE_LABEL)
        logprobs = torch.log_softmax(shifted_logits.float(), dim=-1)
        gathered = logprobs.gather(-1, shifted_input_ids.unsqueeze(-1)).squeeze(-1)
        gathered = gathered.masked_fill(~supervised_mask, 0.0).detach().cpu()

    annotated: list[dict[str, Any]] = []
    for row_index, payload in enumerate(batch):
        length = _payload_length(payload)
        ref_logprobs = [0.0] * length
        if length > 1:
            ref_logprobs[1:] = [float(value) for value in gathered[row_index, : length - 1].tolist()]
        output = dict(payload)
        output["ref_logprobs"] = ref_logprobs
        annotated.append(output)
    return annotated


def _annotate_file(input_path: Path, output_path: Path, model: Any, pad_token_id: int, max_batch_tokens: int) -> dict[str, Any]:
    payloads = list(_iter_msgpack(input_path))

    def annotated_payloads() -> Iterable[dict[str, Any]]:
        for batch in _iter_batches(payloads, max_batch_tokens):
            yield from _annotate_batch(batch, model=model, pad_token_id=pad_token_id)

    count = _write_msgpack(output_path, annotated_payloads())
    return {"input_path": str(input_path), "output_path": str(output_path), "trajectories": count}


def main() -> None:
    args = _parse_args()
    input_dir = Path(args.input_dir)
    assert input_dir.is_dir(), f"input-dir does not exist: {input_dir}"
    files = _trajectory_files(input_dir)
    assert files, f"no trajectory msgpack files found under {input_dir}"
    if args.login_smoke:
        print(json.dumps({"login_smoke": True, "input_dir": str(input_dir), "files": len(files)}), flush=True)
        return

    tokenizer, model = _load_model(args.model_path_or_name, args.fallback_model_name)
    pad_token_id = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
    stats = []
    for input_path, output_path in files:
        stats.append(_annotate_file(input_path, output_path, model, pad_token_id, args.max_batch_tokens))
    print(json.dumps({"annotated_files": stats}, indent=2), flush=True)


if __name__ == "__main__":
    main()
