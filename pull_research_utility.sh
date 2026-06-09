set -euo pipefail

git submodule update --init --recursive research-utility
git submodule update --remote --merge research-utility
