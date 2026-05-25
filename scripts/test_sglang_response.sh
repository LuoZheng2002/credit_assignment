#!/usr/bin/env bash
set -euo pipefail

PORT="${SGLANG_PORT:-30000}"
URL="${SGLANG_URL:-http://localhost:${PORT}/generate}"

response="$(curl -sS "$URL" \
  -H "Content-Type: application/json" \
  --data-raw '{
    "input_ids": [51, 32114],
    "sampling_params": {
      "temperature": 0,
      "max_new_tokens": 10
    },
    "return_logprob": true,
    "logprob_start_len": 0,
    "top_logprobs_num": 5,
    "token_ids_logprob": [[0, 1], [2, 3]],
    "stream": false
  }')"

if command -v jq >/dev/null 2>&1; then
  printf '%s\n' "$response" | jq .
else
  printf '%s\n' "$response"
fi
