source .env

uv run vllm serve Qwen/Qwen3-4B \
  --host 0.0.0.0 \
  --port 8001
