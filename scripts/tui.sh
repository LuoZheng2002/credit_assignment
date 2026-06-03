#!/usr/bin/env bash

set -euo pipefail

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <hostname> <port>"
    exit 1
fi

hostname="$1"
port="$2"

cd "../research-utility"

cargo run --bin bin_progress_tui -- \
    --addr "$hostname:$port"
