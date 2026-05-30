MODEL="${SGLANG_MODEL:-Qwen/Qwen3.5-0.8B}"
HOST="${SGLANG_HOST:-0.0.0.0}"
PORT="${SGLANG_PORT:-30000}"
CONTEXT_LENGTH="${SGLANG_CONTEXT_LENGTH:-8000}"

uv run --project pyprojects/sglang python -m sglang.launch_server \
  --model-path "results/qwen3.5-0.8b/tra16/epoch_0/model" \
  --host "$HOST" \
  --port "$PORT" \
  --context-length "$CONTEXT_LENGTH" \
  "$@"
