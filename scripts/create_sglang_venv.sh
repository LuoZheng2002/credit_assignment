set -euo pipefail

module add python/3.13.5-gcc13.3.1
python -m venv .venv_sglang
source .venv_sglang/bin/activate
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install sglang=0.5.3