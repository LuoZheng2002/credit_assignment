import argparse
import ast
import contextlib
import io
import json
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import scipy
import sympy as sp
from sympy import *  # noqa: F403


class _NullWriter:
    def write(self, _text: str) -> int:
        return 0

    def flush(self) -> None:
        return None


class _ExecutionTimeoutError(Exception):
    pass


MAX_TOOL_OUTPUT_CHARS = 2000


def _load_dotenv_if_present(dotenv_path: str = ".env") -> None:
    path = Path(dotenv_path)
    if not path.exists() or not path.is_file():
        return
    from dotenv import load_dotenv

    load_dotenv(dotenv_path=path, override=False)


def create_request_namespace() -> dict[str, Any]:
    namespace: dict[str, Any] = {
        "np": np,
        "scipy": scipy,
        "sp": sp,
    }
    for symbol_name in sp.__all__:
        namespace[symbol_name] = getattr(sp, symbol_name)
    return namespace


def parent_alive(parent_pid: int) -> bool:
    if parent_pid <= 1:
        return False
    try:
        os.kill(parent_pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def parent_watchdog(parent_pid: int, stop_event: threading.Event) -> None:
    while not stop_event.is_set():
        if not parent_alive(parent_pid):
            os._exit(0)
        stop_event.wait(1.0)


def _timeout_handler(_signum: int, _frame: Any) -> None:
    raise _ExecutionTimeoutError("Python code execution timed out.")


def execute_with_trailing_expression(
    code_text: str, namespace: dict[str, Any], timeout_ms: int
) -> str:
    buf = io.StringIO()
    stderr_sink = _NullWriter()
    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, max(timeout_ms, 1) / 1000.0)
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(stderr_sink):
            tree = ast.parse(code_text, mode="exec")
            if len(tree.body) == 0:
                return ""

            last_stmt = tree.body[-1]
            if isinstance(last_stmt, ast.Expr):
                prefix_module = ast.Module(body=tree.body[:-1], type_ignores=[])
                ast.fix_missing_locations(prefix_module)
                if len(prefix_module.body) > 0:
                    exec(compile(prefix_module, "<tool>", "exec"), namespace, namespace)

                expr = ast.Expression(last_stmt.value)
                ast.fix_missing_locations(expr)
                expr_value = eval(compile(expr, "<tool>", "eval"), namespace, namespace)
                if expr_value is not None:
                    print(repr(expr_value))
            else:
                exec(compile(tree, "<tool>", "exec"), namespace, namespace)
        return buf.getvalue()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


def format_limited_output(output: str, max_chars: int) -> str:
    if max_chars <= 0:
        raise ValueError("max_chars must be greater than zero")
    output_len = len(output)
    if output_len <= max_chars:
        return output

    truncated = output[:max_chars]
    omitted_len = output_len - max_chars
    return (
        f"{truncated}\n"
        "[Output truncated: "
        f"original_length={output_len}, shown={max_chars}, omitted={omitted_len}]"
    )


def main() -> int:
    _load_dotenv_if_present()

    parser = argparse.ArgumentParser(description="Python tool call server")
    parser.add_argument("--parent-pid", type=int, required=True)
    parser.add_argument("--worker-id", type=int, required=False, default=0)
    args = parser.parse_args()

    stop_event = threading.Event()
    watchdog = threading.Thread(
        target=parent_watchdog,
        args=(args.parent_pid, stop_event),
        daemon=True,
    )
    watchdog.start()

    try:
        for raw_line in sys.stdin:
            line = raw_line.strip()
            if line == "":
                continue
            try:
                request = json.loads(line)
                request_id = int(request.get("id", 0))
                code = str(request.get("code", ""))
                timeout_ms = int(request.get("timeout_ms", 5000))
            except Exception as error:  # noqa: BLE001
                response = {
                    "id": 0,
                    "ok": False,
                    "error": f"Malformed request: {error}",
                }
                sys.stdout.write(json.dumps(response) + "\n")
                sys.stdout.flush()
                continue

            try:
                namespace = create_request_namespace()
                output = execute_with_trailing_expression(code, namespace, timeout_ms)
                output = format_limited_output(output, MAX_TOOL_OUTPUT_CHARS)
                response = {"id": request_id, "ok": True, "output": output}
            except _ExecutionTimeoutError as error:
                response = {"id": request_id, "ok": False, "error": str(error)}
            except Exception as error:  # noqa: BLE001
                response = {"id": request_id, "ok": False, "error": str(error)}

            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()
    finally:
        stop_event.set()
        watchdog.join(timeout=0.1)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
