from __future__ import annotations

import atexit
import threading
from typing import Any

from research_utility.tui_message import (
    UnixTuiForwarder,
    delete_worker_bar_message,
    key_value_pair_message,
    master_progress_message,
    worker_progress_message,
)

_TUI_FORWARDER_LOCK = threading.Lock()
_TUI_FORWARDER: UnixTuiForwarder | None = None


def configure_tui_forwarder(socket_path: str | None) -> None:
    global _TUI_FORWARDER

    normalized_socket_path = (socket_path or "").strip()
    with _TUI_FORWARDER_LOCK:
        if _TUI_FORWARDER is not None:
            _TUI_FORWARDER.close()
            _TUI_FORWARDER = None
        if normalized_socket_path:
            _TUI_FORWARDER = UnixTuiForwarder(normalized_socket_path)


def shutdown_tui_forwarder() -> None:
    global _TUI_FORWARDER

    with _TUI_FORWARDER_LOCK:
        if _TUI_FORWARDER is not None:
            _TUI_FORWARDER.close()
            _TUI_FORWARDER = None


def _emit_tui_message(payload: dict[str, Any]) -> bool:
    """Try to send a TUI message via the forwarder. Returns True if forwarded."""
    with _TUI_FORWARDER_LOCK:
        if _TUI_FORWARDER is not None:
            _TUI_FORWARDER.send_message(payload)
            return True
    return False


def send_tui_message(payload: dict[str, Any]) -> None:
    _emit_tui_message(payload)


def _tui_info(message: str) -> None:
    with _TUI_FORWARDER_LOCK:
        if _TUI_FORWARDER is not None:
            _TUI_FORWARDER.send_info(message)


def _tui_warning(message: str) -> None:
    with _TUI_FORWARDER_LOCK:
        if _TUI_FORWARDER is not None:
            _TUI_FORWARDER.send_warning(message)


def _tui_error(message: str) -> None:
    with _TUI_FORWARDER_LOCK:
        if _TUI_FORWARDER is not None:
            _TUI_FORWARDER.send_error(message)


def _tui_key_value(key: str, value: object) -> None:
    send_tui_message(key_value_pair_message(str(key), str(value)))


def _tui_worker_progress(worker_name: str, progress: float, label: str) -> None:
    send_tui_message(
        worker_progress_message(str(worker_name), float(progress), str(label))
    )


def _tui_master_progress(progress: float, label: str) -> None:
    send_tui_message(master_progress_message(float(progress), str(label)))


def _tui_delete_worker_bar(worker_name: str) -> None:
    send_tui_message(delete_worker_bar_message(str(worker_name)))


atexit.register(shutdown_tui_forwarder)
