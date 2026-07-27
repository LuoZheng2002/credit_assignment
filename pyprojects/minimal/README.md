# Minimal Python Environment

This environment is intentionally lightweight for utilities that do not need
CUDA, PyTorch, vLLM, SGLang, or other GPU runtime dependencies. It is used for
paper utilities, Modal submission/download helpers, dataset preparation,
tokenizer downloads, and the Python tool sandbox.

Do not add CUDA-enabled packages or GPU inference/training runtimes here. Keep
those dependencies in the root project or dedicated runtime environments.

Example:

```sh
uv run --project pyprojects/minimal python credit_assignment_paper/scripts/plot_epoch_accuracy.py
```
