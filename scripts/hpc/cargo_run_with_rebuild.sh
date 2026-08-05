#!/bin/bash
# Run a Rust binary through cargo so source changes are respected. If compilation
# fails due to build-cache/filesystem contention, retry once with a clean isolated
# target directory instead of deleting the shared target used by other jobs.

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: scripts/hpc/cargo_run_with_rebuild.sh <bin-name> [args...]" >&2
    exit 2
fi

BIN_NAME="$1"
shift

LOG_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}/slurm/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/cargo_${SLURM_JOB_ID:-manual}_${BIN_NAME}.log"

run_cargo() {
    cargo run --release --bin "$BIN_NAME" -- "$@"
}

set +e
run_cargo "$@" 2> >(tee "$LOG_FILE" >&2)
STATUS=$?
set -e

if [ "$STATUS" -eq 0 ]; then
    exit 0
fi

if ! grep -Eiq 'could not compile|rustc-LLVM ERROR|error: linking with|database is locked|stale file handle|failed to save last-use data' "$LOG_FILE"; then
    exit "$STATUS"
fi

RETRY_TARGET_PARENT="${SLURM_TMPDIR:-/tmp}/${USER:-user}/credit_assignment_cargo_retry"
RETRY_TARGET_DIR="$RETRY_TARGET_PARENT/${SLURM_JOB_ID:-manual}_${BIN_NAME}"
rm -rf "$RETRY_TARGET_DIR"
mkdir -p "$RETRY_TARGET_DIR"

echo "cargo run failed with a compile/cache-like error; retrying clean isolated build at $RETRY_TARGET_DIR" >&2
CARGO_TARGET_DIR="$RETRY_TARGET_DIR" cargo run --release --bin "$BIN_NAME" -- "$@"
