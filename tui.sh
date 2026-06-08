#!/usr/bin/env bash

set -euo pipefail

if [ "$#" -gt 1 ]; then
    echo "Usage: $0 [progress_log_file]"
    exit 1
fi

# script_dir="$(cd "$(dirname "$0")" && pwd)"
# log_file="${1:-$script_dir/progress_tui_log.bin}"

log_file="$(realpath "$1")"

# cd "$script_dir/../research-utility"
cd ../research-utility

cargo run --bin bin_progress_tui -- \
    --log-file "$log_file"
