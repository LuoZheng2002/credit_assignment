import argparse
import ast
import builtins
import contextlib
import io
import json
import os
import shutil
import signal
import socket
import sys
import tempfile
import time
from multiprocessing import get_context
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any

import numpy as np
import scipy
import sympy as sp

MAX_TOOL_OUTPUT_CHARS = 2000
DEFAULT_REQUEST_TIMEOUT_MS = 1000
_CURRENT_REQUEST_TIMEOUT_MS = DEFAULT_REQUEST_TIMEOUT_MS
_REQUEST_CONTEXT = get_context("fork")


def python_request_timeout_error_message(timeout_ms: int) -> str:
    return f"Python code execution timed out after {timeout_ms} ms."


class SandboxViolation(PermissionError):
    pass


class RequestTimedOut(TimeoutError):
    pass


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


_BASE_REQUEST_NAMESPACE = create_request_namespace()


def execute_with_trailing_expression(code_text: str, namespace: dict[str, Any]) -> str:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
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


def execute_single_request(code: str) -> dict[str, Any]:
    try:
        namespace = dict(_BASE_REQUEST_NAMESPACE)
        output = execute_with_trailing_expression(code, namespace)
        output = format_limited_output(output, MAX_TOOL_OUTPUT_CHARS)
        return {"ok": True, "output": output}
    except Exception as error:  # noqa: BLE001
        return {"ok": False, "error": str(error)}


_ORIGINAL_OPEN = builtins.open
_ORIGINAL_IO_OPEN = io.open
_ORIGINAL_OS_OPEN = os.open
_ORIGINAL_PATH_OPEN = Path.open


def _deny(reason: str) -> None:
    raise SandboxViolation(reason)


def _mode_is_write(mode: str) -> bool:
    return any(flag in mode for flag in ("w", "a", "x", "+"))


def _flags_allow_write(flags: int) -> bool:
    write_flags = 0
    for flag_name in ("O_WRONLY", "O_RDWR", "O_APPEND", "O_CREAT", "O_TRUNC", "O_EXCL"):
        write_flags |= getattr(os, flag_name, 0)
    return (flags & write_flags) != 0


def _guarded_open(file: Any, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
    if _mode_is_write(mode):
        _deny("sandbox forbids writing files to disk")
    return _ORIGINAL_OPEN(file, mode, *args, **kwargs)


def _guarded_io_open(file: Any, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
    if _mode_is_write(mode):
        _deny("sandbox forbids writing files to disk")
    return _ORIGINAL_IO_OPEN(file, mode, *args, **kwargs)


def _guarded_os_open(path: Any, flags: int, *args: Any, **kwargs: Any) -> int:
    if _flags_allow_write(flags):
        _deny("sandbox forbids writing files to disk")
    return _ORIGINAL_OS_OPEN(path, flags, *args, **kwargs)


def _guarded_path_open(self: Path, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
    if _mode_is_write(mode):
        _deny("sandbox forbids writing files to disk")
    return _ORIGINAL_PATH_OPEN(self, mode, *args, **kwargs)


def _deny_file_mutation(*args: Any, **kwargs: Any) -> Any:
    del args, kwargs
    _deny("sandbox forbids mutating the filesystem")


def _deny_process_creation(*args: Any, **kwargs: Any) -> Any:
    del args, kwargs
    _deny("sandbox forbids launching subprocesses")


def _deny_network(*args: Any, **kwargs: Any) -> Any:
    del args, kwargs
    _deny("sandbox forbids network access")


def _install_file_write_guardrails() -> None:
    builtins.open = _guarded_open
    io.open = _guarded_io_open
    os.open = _guarded_os_open
    Path.open = _guarded_path_open
    Path.write_text = _deny_file_mutation
    Path.write_bytes = _deny_file_mutation
    Path.touch = _deny_file_mutation
    Path.mkdir = _deny_file_mutation
    Path.rename = _deny_file_mutation
    Path.replace = _deny_file_mutation
    Path.unlink = _deny_file_mutation
    Path.rmdir = _deny_file_mutation
    Path.chmod = _deny_file_mutation
    if hasattr(Path, "lchmod"):
        Path.lchmod = _deny_file_mutation
    if hasattr(Path, "symlink_to"):
        Path.symlink_to = _deny_file_mutation
    if hasattr(Path, "hardlink_to"):
        Path.hardlink_to = _deny_file_mutation

    os.remove = _deny_file_mutation
    os.unlink = _deny_file_mutation
    os.rename = _deny_file_mutation
    os.replace = _deny_file_mutation
    os.mkdir = _deny_file_mutation
    os.makedirs = _deny_file_mutation
    os.rmdir = _deny_file_mutation
    os.removedirs = _deny_file_mutation
    os.chmod = _deny_file_mutation
    if hasattr(os, "lchmod"):
        os.lchmod = _deny_file_mutation
    if hasattr(os, "chown"):
        os.chown = _deny_file_mutation
    if hasattr(os, "lchown"):
        os.lchown = _deny_file_mutation
    os.utime = _deny_file_mutation

    shutil.copy = _deny_file_mutation
    shutil.copy2 = _deny_file_mutation
    shutil.copyfile = _deny_file_mutation
    shutil.copytree = _deny_file_mutation
    shutil.move = _deny_file_mutation
    shutil.rmtree = _deny_file_mutation

    tempfile.TemporaryFile = _deny_file_mutation
    tempfile.NamedTemporaryFile = _deny_file_mutation
    tempfile.SpooledTemporaryFile = _deny_file_mutation
    tempfile.mkstemp = _deny_file_mutation
    tempfile.mkdtemp = _deny_file_mutation


def _install_process_and_network_guardrails() -> None:
    os.system = _deny_process_creation
    if hasattr(os, "popen"):
        os.popen = _deny_process_creation
    for name in (
        "spawnl",
        "spawnle",
        "spawnlp",
        "spawnlpe",
        "spawnv",
        "spawnve",
        "spawnvp",
        "spawnvpe",
    ):
        if hasattr(os, name):
            setattr(os, name, _deny_process_creation)

    try:
        import subprocess  # noqa: PLC0415

        subprocess.Popen = _deny_process_creation
        subprocess.run = _deny_process_creation
        subprocess.call = _deny_process_creation
        subprocess.check_call = _deny_process_creation
        subprocess.check_output = _deny_process_creation
    except Exception:  # noqa: BLE001
        pass

    socket.socket = _deny_network
    socket.create_connection = _deny_network
    if hasattr(socket, "fromfd"):
        socket.fromfd = _deny_network
    if hasattr(socket, "socketpair"):
        socket.socketpair = _deny_network


def _apply_resource_limits() -> None:
    try:
        import resource  # noqa: PLC0415

        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        if hasattr(resource, "RLIMIT_FSIZE"):
            resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
    except Exception:  # noqa: BLE001
        pass


def install_request_sandbox() -> None:
    sys.dont_write_bytecode = True
    _install_file_write_guardrails()
    _install_process_and_network_guardrails()
    _apply_resource_limits()


def _request_timeout_handler(_: int, __: Any) -> None:
    raise RequestTimedOut(
        python_request_timeout_error_message(_CURRENT_REQUEST_TIMEOUT_MS)
    )


def _run_request_in_child(connection: Connection, code: str, timeout_ms: int) -> None:
    global _CURRENT_REQUEST_TIMEOUT_MS

    response: dict[str, Any] = {
        "ok": False,
        "error": "Python request worker failed before producing a response.",
    }
    try:
        if hasattr(os, "setsid"):
            os.setsid()
        _CURRENT_REQUEST_TIMEOUT_MS = timeout_ms
        install_request_sandbox()
        signal.signal(signal.SIGALRM, _request_timeout_handler)
        signal.setitimer(signal.ITIMER_REAL, timeout_ms / 1000.0)
        response = execute_single_request(code)
    except BaseException as error:  # noqa: BLE001
        response = {"ok": False, "error": str(error)}
    finally:
        try:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
        except Exception:  # noqa: BLE001
            pass
        try:
            connection.send(response)
        except Exception:  # noqa: BLE001
            pass
        connection.close()


def _kill_process_group(pid: int) -> None:
    if pid <= 0:
        return
    if hasattr(os, "killpg"):
        try:
            os.killpg(pid, signal.SIGKILL)
            return
        except ProcessLookupError:
            return
        except Exception:  # noqa: BLE001
            pass
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def execute_request_with_timeout(code: str, timeout_ms: int) -> dict[str, Any]:
    parent_conn, child_conn = _REQUEST_CONTEXT.Pipe(duplex=False)
    worker = _REQUEST_CONTEXT.Process(
        target=_run_request_in_child,
        args=(child_conn, code, timeout_ms),
    )
    worker.start()
    child_conn.close()

    response: dict[str, Any] | None = None
    deadline = time.monotonic() + (timeout_ms / 1000.0)
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            if parent_conn.poll(remaining):
                try:
                    candidate = parent_conn.recv()
                except EOFError:
                    response = {
                        "ok": False,
                        "error": "Python request worker exited without returning a response.",
                    }
                else:
                    if isinstance(candidate, dict):
                        response = candidate
                    else:
                        response = {
                            "ok": False,
                            "error": "Python request worker returned a malformed response.",
                        }
                break

        if response is None:
            _kill_process_group(worker.pid or -1)
            worker.kill()
            worker.join(timeout=0.2)
            return {
                "ok": False,
                "error": python_request_timeout_error_message(timeout_ms),
            }

        worker.join(timeout=0.2)
        if worker.is_alive():
            _kill_process_group(worker.pid or -1)
            worker.kill()
            worker.join(timeout=0.2)
        return response
    finally:
        parent_conn.close()


def serve_forever(request_timeout_ms: int) -> int:
    sys.stdout.write(json.dumps({"ready": True}) + "\n")
    sys.stdout.flush()

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError as error:
            response = {"ok": False, "error": f"Malformed tool request JSON: {error}"}
        else:
            code = request.get("code")
            if not isinstance(code, str):
                response = {
                    "ok": False,
                    "error": "Malformed tool request: expected string field 'code'.",
                }
            else:
                response = execute_request_with_timeout(code, request_timeout_ms)

        sys.stdout.write(json.dumps(response) + "\n")
        sys.stdout.flush()

    return 0


def main() -> int:
    _load_dotenv_if_present()

    parser = argparse.ArgumentParser(description="Python tool executor")
    parser.add_argument("--single-shot", action="store_true")
    parser.add_argument("--persistent-server", action="store_true")
    parser.add_argument(
        "--request-timeout-ms",
        type=int,
        default=DEFAULT_REQUEST_TIMEOUT_MS,
    )
    args = parser.parse_args()

    if args.request_timeout_ms <= 0:
        sys.stderr.write("--request-timeout-ms must be greater than zero\n")
        return 2

    if args.single_shot == args.persistent_server:
        sys.stderr.write(
            "exactly one of --single-shot or --persistent-server must be set\n"
        )
        return 2

    if args.single_shot:
        code = sys.stdin.read()
        response = execute_request_with_timeout(code, args.request_timeout_ms)
        sys.stdout.write(json.dumps(response) + "\n")
        sys.stdout.flush()
        return 0

    return serve_forever(args.request_timeout_ms)


if __name__ == "__main__":
    raise SystemExit(main())
