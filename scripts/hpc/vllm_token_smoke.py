#!/usr/bin/env python3
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def _post_json(url: str, payload: dict, timeout: float = 30.0) -> tuple[int, str]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode("utf-8", errors="replace")


def _get(url: str, timeout: float = 5.0) -> tuple[int, str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.status, response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode("utf-8", errors="replace")
    except OSError as error:
        return 0, str(error)


def _find_token_id_fields(value):
    fields = []
    if isinstance(value, dict):
        for key, item in value.items():
            if "token" in key.lower() and "id" in key.lower():
                fields.append((key, item))
            fields.extend(_find_token_id_fields(item))
    elif isinstance(value, list):
        for item in value:
            fields.extend(_find_token_id_fields(item))
    return fields


def _wait_for_server(base_url: str, process: subprocess.Popen, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last_status = ""
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"vLLM server exited early with status {process.returncode}")
        status, body = _get(f"{base_url}/health")
        last_status = f"{status} {body[:200]}"
        if status == 200:
            return
        time.sleep(5)
    raise RuntimeError(f"vLLM server did not become healthy; last status: {last_status}")


def _make_prompt_token_ids(model_path: Path) -> list[int]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    token_ids = tokenizer.encode("Solve 1+1. Answer:", add_special_tokens=False)
    if not token_ids:
        raise RuntimeError("Tokenizer produced an empty prompt")
    return token_ids


def main() -> int:
    model_path = Path(
        os.environ.get(
            "VLLM_SMOKE_MODEL",
            "/work/nvme/bhph/zluo8/credit_assignment/results/large_files/qwen25/model",
        )
    )
    port = int(os.environ.get("VLLM_SMOKE_PORT", "18080"))
    startup_timeout = float(os.environ.get("VLLM_SMOKE_STARTUP_TIMEOUT", "900"))
    served_model_name = os.environ.get("VLLM_SMOKE_SERVED_MODEL_NAME", "vllm-smoke")

    if not model_path.exists():
        raise RuntimeError(f"Model path does not exist: {model_path}")

    import vllm

    print(f"vllm_version={vllm.__version__}", flush=True)
    print(f"model_path={model_path}", flush=True)
    prompt_token_ids = _make_prompt_token_ids(model_path)
    print(f"prompt_token_ids={prompt_token_ids}", flush=True)

    command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        str(model_path),
        "--served-model-name",
        served_model_name,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--skip-tokenizer-init",
        "--dtype",
        "auto",
        "--gpu-memory-utilization",
        os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", "0.70"),
        "--max-model-len",
        os.environ.get("VLLM_MAX_MODEL_LEN", "4096"),
        "--max-num-seqs",
        os.environ.get("VLLM_MAX_NUM_SEQS", "4"),
        "--max-num-batched-tokens",
        os.environ.get("VLLM_MAX_NUM_BATCHED_TOKENS", "4096"),
        "--enforce-eager",
    ]
    print("server_command=" + " ".join(command), flush=True)

    process = subprocess.Popen(command)
    base_url = f"http://127.0.0.1:{port}"
    try:
        _wait_for_server(base_url, process, startup_timeout)
        payload = {
            "model": served_model_name,
            "prompt": prompt_token_ids,
            "max_tokens": 8,
            "temperature": 0,
            "logprobs": 5,
            "return_token_ids": True,
            "return_tokens_as_token_ids": True,
        }
        status, body = _post_json(f"{base_url}/v1/completions", payload)
        print(f"completion_status={status}", flush=True)
        print(f"completion_body={body}", flush=True)
        if status != 200:
            return 2
        response = json.loads(body)
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("Completion response did not contain choices")
        token_id_fields = _find_token_id_fields(response)
        print(f"token_id_fields={json.dumps(token_id_fields, ensure_ascii=False)}", flush=True)
        if not token_id_fields:
            raise RuntimeError("Completion response did not expose token id fields")
        return 0
    finally:
        if process.poll() is None:
            process.send_signal(signal.SIGTERM)
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=30)


if __name__ == "__main__":
    raise SystemExit(main())
