source activate_environment.sh


vllm serve Qwen/Qwen3-4B \
  --host 0.0.0.0 \
  --port 8001 \
  --tensor-parallel-size 4
