#!/bin/bash

if command -v module >/dev/null 2>&1; then
    module unload cudatoolkit >/dev/null 2>&1 || true
    if module --ignore_cache load cudatoolkit/25.3_12.8 >/dev/null 2>&1; then
        :
    elif module --ignore_cache load cudatoolkit/26.5_13.2 >/dev/null 2>&1; then
        :
    else
        module --ignore_cache load cudatoolkit/25.3_11.8
    fi
fi

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
export CUDA_PATH="${CUDA_PATH:-$CUDA_HOME}"
export PATH="$CUDA_HOME/bin:$PATH"
export LIBRARY_PATH="/lib64:$CUDA_HOME/lib64:$CUDA_HOME/targets/x86_64-linux/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export CPATH="$CUDA_HOME/include:$CUDA_HOME/targets/x86_64-linux/include${CPATH:+:$CPATH}"
