# #!/usr/bin/env bash
# set -euo pipefail

# SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# PY_SCRIPT="${SCRIPT_DIR}/../research-utility/src_py/sqlite_to_jsonl_all.py"

# python3 "${PY_SCRIPT}" --repo-root "${SCRIPT_DIR}" "$@"
source activate_environment.sh

python -m research_utility.sqlite_to_jsonl_all --repo-root . "$@"
