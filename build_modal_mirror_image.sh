#!/usr/bin/env bash
set -euo pipefail

IMAGE_TAG="${1:-credit-assignment:modal-mirror}"

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is not installed or not in PATH" >&2
  exit 1
fi

if docker buildx version >/dev/null 2>&1; then
  docker buildx build \
    --load \
    -f Dockerfile.modal-mirror \
    -t "${IMAGE_TAG}" \
    .
else
  docker build \
    -f Dockerfile.modal-mirror \
    -t "${IMAGE_TAG}" \
    .
fi

echo "Built image: ${IMAGE_TAG}"
