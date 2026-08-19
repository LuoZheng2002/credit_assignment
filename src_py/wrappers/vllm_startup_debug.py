"""Timestamped launcher for debugging vLLM startup latency."""

from __future__ import annotations

import os
import runpy
import sys
import time
from datetime import datetime, timezone


def _log(message: str) -> None:
    timestamp = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    print(
        f"[VLLM_STARTUP_DEBUG] ts={timestamp} t={time.time():.3f} pid={os.getpid()} {message}",
        flush=True,
    )


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
    _log("before_run_module")
    sys.argv = [module_name, *module_args]
    runpy.run_module(module_name, run_name="__main__", alter_sys=True)
    _log("run_module_returned")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
