from __future__ import annotations

import atexit
import builtins
import json
import sys
import threading
from pathlib import Path
from typing import Any

from research_utility.tui_message import UnixTuiForwarder

_ORIGINAL_PRINT = builtins.print
_BUFFER_LOCK = threading.Lock()
_BUFFERED_LINES: list[str] = []
_STOP_EVENT = threading.Event()
_POLL_THREAD: threading.Thread | None = None
_TUI_FORWARDER: UnixTuiForwarder | None = None
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
    if _TUI_FORWARDER is None:
        return
    stripped = line.strip()
    if not stripped:
        return

    try:
        payload = json.loads(stripped)
        if _looks_like_tui_message_payload(payload):
            _TUI_FORWARDER.send_message(payload)
            return
    except json.JSONDecodeError:
        pass

    if stripped.startswith("[error] "):
        _TUI_FORWARDER.send_error(stripped[len("[error] ") :])
        return
    if stripped.startswith("[warning] "):
        _TUI_FORWARDER.send_warning(stripped[len("[warning] ") :])
        return
    if stripped.startswith("[status] "):
        _TUI_FORWARDER.send_info(stripped[len("[status] ") :])
        return
    if stripped.startswith("[startup] "):
        _TUI_FORWARDER.send_info(stripped[len("[startup] ") :])
        return

    if file is sys.stderr:
        _TUI_FORWARDER.send_warning(stripped)
    else:
        _TUI_FORWARDER.send_info(stripped)


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


def _flush_poller(job_folder: Path) -> None:
    request_file = job_folder / ".flush_logs.request"
    while not _STOP_EVENT.is_set():
        if request_file.exists():
            try:
                request_file.unlink()
            except FileNotFoundError:
                pass
            flush_buffered_lines()
        _STOP_EVENT.wait(0.2)


def install_status_log_buffer(
    job_folder_path: str, orchestrator_socket_path: str = ""
) -> None:
    global _POLL_THREAD, _TUI_FORWARDER

    job_folder = Path(job_folder_path)
    job_folder.mkdir(parents=True, exist_ok=True)

    builtins.print = buffered_print
    _STOP_EVENT.clear()
    _TUI_FORWARDER = UnixTuiForwarder(orchestrator_socket_path)

    if _POLL_THREAD is None or not _POLL_THREAD.is_alive():
        _POLL_THREAD = threading.Thread(
            target=_flush_poller, args=(job_folder,), daemon=True
        )
        _POLL_THREAD.start()


def shutdown_status_log_buffer() -> None:
    global _TUI_FORWARDER
    _STOP_EVENT.set()
    flush_buffered_lines()
    builtins.print = _ORIGINAL_PRINT
    if _TUI_FORWARDER is not None:
        _TUI_FORWARDER.close()
        _TUI_FORWARDER = None


atexit.register(shutdown_status_log_buffer)
