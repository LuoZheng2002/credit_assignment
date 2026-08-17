#!/bin/bash

if command -v module >/dev/null 2>&1; then
    module unload cudatoolkit >/dev/null 2>&1 || true
    module load cudatoolkit/25.3_12.8
fi

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
export CUDA_PATH="${CUDA_PATH:-$CUDA_HOME}"
export PATH="$CUDA_HOME/bin:$PATH"
export LIBRARY_PATH="/lib64:$CUDA_HOME/lib64:$CUDA_HOME/targets/x86_64-linux/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export CPATH="$CUDA_HOME/include:$CUDA_HOME/targets/x86_64-linux/include${CPATH:+:$CPATH}"
