#!/usr/bin/env python3
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "openai/gpt-4o"


def read_env_file_key() -> str | None:
    root_env = Path(__file__).resolve().parents[1] / ".env"
    if not root_env.exists():
        return None
    for raw_line in root_env.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() != "OPENROUTER_API_KEY":
            continue
        value = value.strip().strip('"').strip("'")
        return value or None
    return None


def get_api_key() -> str:
    api_key = os.getenv("OPENROUTER_API_KEY") or read_env_file_key()
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not found in env or .env")
    return api_key


def http_json(method: str, url: str, api_key: str, payload: dict | None = None) -> dict:
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url=url, method=method, data=data)
    req.add_header("Authorization", f"Bearer {api_key}")
    req.add_header("Content-Type", "application/json")
    req.add_header("Accept", "application/json")

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as err:
        body = err.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {err.code} for {url}: {body}") from err


def list_providers(api_key: str, model: str) -> list[str]:
    url = f"{OPENROUTER_BASE_URL}/models/{model}/endpoints"
    data = http_json("GET", url, api_key)
    providers: list[str] = []

    endpoints = data.get("data", {}).get("endpoints", [])
    for endpoint in endpoints:
        provider_name = endpoint.get("provider_name")
        if isinstance(provider_name, str) and provider_name not in providers:
            providers.append(provider_name)
    return providers


def test_provider(api_key: str, model: str, provider: str) -> tuple[bool, str]:
    url = f"{OPENROUTER_BASE_URL}/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with one short word."}],
        "max_tokens": 16,
        "logprobs": True,
        "top_logprobs": 3,
        "provider": {
            "order": [provider],
            "allow_fallbacks": False,
            "require_parameters": True,
        },
    }

    try:
        data = http_json("POST", url, api_key, payload)
    except Exception as err:  # noqa: BLE001
        return False, str(err)

    choice0 = (data.get("choices") or [{}])[0]
    logprobs = choice0.get("logprobs")
    if logprobs is None:
        return False, "response has choices[0].logprobs = null"

    content = logprobs.get("content") if isinstance(logprobs, dict) else None
    if not isinstance(content, list):
        return False, "response missing choices[0].logprobs.content"

    return True, "ok"


def main() -> int:
    model = DEFAULT_MODEL
    if len(sys.argv) > 1:
        model = sys.argv[1]

    api_key = get_api_key()

    providers = list_providers(api_key, model)
    if not providers:
        print(f"No providers found for model {model}")
        return 2

    print(f"Testing {len(providers)} providers for {model}...")
    passing: list[str] = []

    for provider in providers:
        ok, detail = test_provider(api_key, model, provider)
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {provider}: {detail}")
        if ok:
            passing.append(provider)

    print()
    if passing:
        print("Providers that support require_parameters=true with logprobs:")
        for provider in passing:
            print(f"- {provider}")
        return 0

    print("No provider passed this check.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
