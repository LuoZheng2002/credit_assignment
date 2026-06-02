# #!/usr/bin/env bash
# set -euo pipefail

MODEL="${SGLANG_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
HOST="${SGLANG_HOST:-0.0.0.0}"
PORT="${SGLANG_PORT:-30000}"
CONTEXT_LENGTH="${SGLANG_CONTEXT_LENGTH:-8192}"

uv run --project pyprojects/sglang python -m sglang.launch_server \
  --model-path "$MODEL" \
  --host "$HOST" \
  --port "$PORT" \
  --context-length "$CONTEXT_LENGTH" \
  --load-balance-method total_tokens \
  --chunked-prefill-size 8192 \
  --enable-mixed-chunk \
  --schedule-policy lpm \
  "$@"


# apptainer run --nv sglang-cu12.sif \
#   python3 -m sglang.launch_server \
#     --model-path Qwen/Qwen2.5-7B-Instruct \
#     --host 0.0.0.0 \
#     --port 30000
