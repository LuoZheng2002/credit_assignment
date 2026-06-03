#!/usr/bin/env bash

set -euo pipefail

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <hostname> <port>"
    exit 1
fi

hostname="$1"
port="$2"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir/../research-utility"

cargo run --bin bin_progress_tui -- \
    --addr "$hostname:$port"
