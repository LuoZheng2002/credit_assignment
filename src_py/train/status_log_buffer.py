from __future__ import annotations

import atexit
import builtins
import json
import sys
import threading
from typing import Any

from ..tui_logging import (
    _tui_error,
    _tui_info,
    _tui_warning,
    configure_tui_forwarder,
    send_tui_message,
    shutdown_tui_forwarder,
)

_ORIGINAL_PRINT = builtins.print
_BUFFER_LOCK = threading.Lock()
_BUFFERED_LINES: list[str] = []

_TUI_MESSAGE_VARIANTS = {
    "Line",
    "State",
    "WindowName",
    "KeyValuePair",
    "WorkerProgress",
    "MasterProgress",
    "DeleteWorkerBar",
    "ExitHint",
}


def _format_line(*args: object, sep: str, end: str) -> str:
    return sep.join(str(arg) for arg in args) + end


def _looks_like_tui_message_payload(payload: Any) -> bool:
    if not isinstance(payload, dict) or len(payload) != 1:
        return False
    [(key, _value)] = payload.items()
    return key in _TUI_MESSAGE_VARIANTS


def _forward_line_to_tui(line: str, *, file: Any) -> None:
    stripped = line.strip()
    if not stripped:
        return

    try:
        payload = json.loads(stripped)
        if _looks_like_tui_message_payload(payload):
            send_tui_message(payload)
            return
    except json.JSONDecodeError:
        pass

    if stripped.startswith("[error] "):
        _tui_error(stripped[len("[error] ") :])
        return
    if stripped.startswith("[warning] "):
        _tui_warning(stripped[len("[warning] ") :])
        return
    if stripped.startswith("[status] "):
        _tui_info(stripped[len("[status] ") :])
        return
    if stripped.startswith("[startup] "):
        _tui_info(stripped[len("[startup] ") :])
        return

    if file is sys.stderr:
        _tui_warning(stripped)
    else:
        _tui_info(stripped)


def buffered_print(
    *args: object, sep: str = " ", end: str = "\n", file=None, flush: bool = False
) -> None:
    if file is not None and file not in {None, sys.stdout, sys.stderr}:
        _ORIGINAL_PRINT(*args, sep=sep, end=end, file=file, flush=flush)
        return

    line = _format_line(*args, sep=sep, end=end)
    with _BUFFER_LOCK:
        _BUFFERED_LINES.append(line)
    _forward_line_to_tui(line, file=file)


def flush_buffered_lines() -> int:
    with _BUFFER_LOCK:
        if len(_BUFFERED_LINES) == 0:
            return 0
        lines = list(_BUFFERED_LINES)
        _BUFFERED_LINES.clear()

    for line in lines:
        _ORIGINAL_PRINT(line, end="")
    return len(lines)


def install_status_log_buffer(orchestrator_socket_path: str = "") -> None:
    builtins.print = buffered_print
    configure_tui_forwarder(orchestrator_socket_path)


def shutdown_status_log_buffer() -> None:
    flush_buffered_lines()
    builtins.print = _ORIGINAL_PRINT
    shutdown_tui_forwarder()


atexit.register(shutdown_status_log_buffer)
