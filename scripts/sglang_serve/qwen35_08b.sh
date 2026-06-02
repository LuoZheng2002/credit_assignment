MODEL="${SGLANG_MODEL:-Qwen/Qwen3.5-0.8B}"
HOST="${SGLANG_HOST:-0.0.0.0}"
PORT="${SGLANG_PORT:-30000}"
CONTEXT_LENGTH="${SGLANG_CONTEXT_LENGTH:-8192}"

uv run --project pyprojects/sglang python -m sglang.launch_server \
  --model-path "$MODEL" \
  --host "$HOST" \
  --port "$PORT" \
  --context-length "$CONTEXT_LENGTH" \
  --load-balance-method total_tokens \
  --chunked-prefill-size "$CONTEXT_LENGTH" \
  --enable-mixed-chunk \
  --schedule-policy lpm \
  "$@"
