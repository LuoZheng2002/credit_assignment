#!/usr/bin/env bash

PROTO_DIR="proto"
OUT_DIR="src_py/vllm_wrapper_proto"

mkdir -p "${OUT_DIR}"

uv run --extra gpu -m grpc_tools.protoc \
  -I"${PROTO_DIR}" \
  --python_out="${OUT_DIR}" \
  --grpc_python_out="${OUT_DIR}" \
  "${PROTO_DIR}/vllm_wrapper.proto"

touch "${OUT_DIR}/__init__.py"
