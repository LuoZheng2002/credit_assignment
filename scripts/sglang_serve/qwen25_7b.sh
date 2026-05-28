# #!/usr/bin/env bash
# set -euo pipefail

MODEL="${SGLANG_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
HOST="${SGLANG_HOST:-0.0.0.0}"
PORT="${SGLANG_PORT:-30000}"
CONTEXT_LENGTH="${SGLANG_CONTEXT_LENGTH:-6000}"

uv run python -m sglang.launch_server \
  --model-path "$MODEL" \
  --host "$HOST" \
  --port "$PORT" \
  --context-length "$CONTEXT_LENGTH" \
  "$@"


# apptainer run --nv sglang-cu12.sif \
#   python3 -m sglang.launch_server \
#     --model-path Qwen/Qwen2.5-7B-Instruct \
#     --host 0.0.0.0 \
#     --port 30000
