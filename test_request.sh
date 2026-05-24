#!/usr/bin/env bash
set -euo pipefail

PORT="${VLLM_PORT:-8000}"
MODEL="${VLLM_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
URL="http://localhost:${PORT}/v1/completions"

response="$(curl -sS "$URL" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"${MODEL}\",\"prompt\":\"Hello\",\"max_tokens\":32,\"logprobs\":8}")"

if command -v jq >/dev/null 2>&1; then
  printf '%s\n' "$response" | jq .
else
  printf '%s\n' "$response"
fi
