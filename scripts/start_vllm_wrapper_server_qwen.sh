# ROOT_DIR="$(git rev-parse --show-toplevel)"

# if [[ -z "${VLLM_WRAPPER_MODEL:-}" ]]; then
#   echo "Usage: VLLM_WRAPPER_MODEL=<model> $0 [additional server args]"
#   echo "Example: VLLM_WRAPPER_MODEL=Qwen/Qwen3-4B $0 --port 50051"
#   exit 1
# fi

# if [[ ! -f "${ROOT_DIR}/src_py/vllm_wrapper_proto/vllm_wrapper_pb2.py" ]] || [[ ! -f "${ROOT_DIR}/src_py/vllm_wrapper_proto/vllm_wrapper_pb2_grpc.py" ]]; then
#   echo "Generated protobuf files not found. Generating..."
#   "${ROOT_DIR}/scripts/gen_vllm_wrapper_proto.sh"
# fi
# cd "${ROOT_DIR}"

uv run --extra gpu python -m src_py.vllm_wrapper.server \
  --host 127.0.0.1 \
  --port 50051 \
  --model "Qwen/Qwen2.5-7B-Instruct" \
  --gpu-memory-utilization 0.9
