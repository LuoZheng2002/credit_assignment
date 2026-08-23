"""Timestamped launcher for debugging vLLM startup latency."""

from __future__ import annotations

import os
import runpy
import sys
import time
import builtins
from datetime import datetime, timezone


def _log(message: str) -> None:
    timestamp = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    print(
        f"[VLLM_STARTUP_DEBUG] ts={timestamp} t={time.time():.3f} pid={os.getpid()} {message}",
        flush=True,
    )


_LARGE_IMPORT_PREFIXES = (
    "vllm",
    "torch",
    "transformers",
    "triton",
    "xformers",
    "flash_attn",
    "numpy",
    "scipy",
    "pydantic",
    "ray",
)


def _install_import_timing() -> None:
    original_import = builtins.__import__
    seen: set[str] = set()

    def timed_import(name, globals=None, locals=None, fromlist=(), level=0):
        root_name = name.split(".", 1)[0]
        should_log = level == 0 and root_name in _LARGE_IMPORT_PREFIXES
        first_seen = should_log and root_name not in seen
        if first_seen:
            seen.add(root_name)
            start = time.time()
            _log(f"large_import_start package={root_name} requested={name}")
            try:
                return original_import(name, globals, locals, fromlist, level)
            finally:
                _log(
                    "large_import_done "
                    f"package={root_name} requested={name} elapsed_secs={time.time() - start:.3f}"
                )
        return original_import(name, globals, locals, fromlist, level)

    builtins.__import__ = timed_import


def main() -> int:
    if len(sys.argv) < 2:
        print(
            "Usage: python -m src_py.wrappers.vllm_startup_debug <module> [args...]",
            file=sys.stderr,
            flush=True,
        )
        return 2

    module_name = sys.argv[1]
    module_args = sys.argv[2:]
    _log(f"launcher_started module={module_name} argv_len={len(module_args)}")
    _log(f"cwd={os.getcwd()}")
    _log(f"python={sys.executable}")
    _log(f"argv={' '.join(sys.argv)}")
    for env_name in (
        "VLLM_VENV",
        "XDG_CACHE_HOME",
        "TRITON_CACHE_DIR",
        "TORCHINDUCTOR_CACHE_DIR",
        "CUDA_CACHE_PATH",
        "VLLM_MAX_NUM_SEQS",
        "VLLM_MAX_NUM_BATCHED_TOKENS",
        "VLLM_ENFORCE_EAGER",
    ):
        _log(f"env {env_name}={os.environ.get(env_name, '')}")
    _install_import_timing()
    _log("before_run_module")
    sys.argv = [module_name, *module_args]
    runpy.run_module(module_name, run_name="__main__", alter_sys=True)
    _log("run_module_returned")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
