# Minimal Python Environment

This environment is intentionally lightweight for local paper utilities such as
plot generation. Do not add CUDA, PyTorch, Transformers, vLLM, SGLang, or other
LLM runtime dependencies here.

Example:

```sh
uv run --project pyprojects/minimal python credit_assignment_paper/scripts/plot_epoch_accuracy.py
```
