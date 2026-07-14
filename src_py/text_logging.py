from __future__ import annotations

import atexit
import threading
from typing import Any

from research_utility.text_message import (
    UnixTextForwarder,
    delete_worker_bar_message,
    key_value_pair_message,
    master_progress_message,
    worker_progress_message,
)

_TEXT_FORWARDER_LOCK = threading.Lock()
_TEXT_FORWARDER: UnixTextForwarder | None = None


def configure_text_forwarder(socket_path: str | None) -> None:
    global _TEXT_FORWARDER

    normalized_socket_path = (socket_path or "").strip()
    with _TEXT_FORWARDER_LOCK:
        if _TEXT_FORWARDER is not None:
            _TEXT_FORWARDER.close()
            _TEXT_FORWARDER = None
        if normalized_socket_path:
            _TEXT_FORWARDER = UnixTextForwarder(normalized_socket_path)


def shutdown_text_forwarder() -> None:
    global _TEXT_FORWARDER

    with _TEXT_FORWARDER_LOCK:
        if _TEXT_FORWARDER is not None:
            _TEXT_FORWARDER.close()
            _TEXT_FORWARDER = None


def _emit_text_message(payload: dict[str, Any]) -> bool:
    """Try to send a text message via the forwarder. Returns True if forwarded."""
    with _TEXT_FORWARDER_LOCK:
        if _TEXT_FORWARDER is not None:
            _TEXT_FORWARDER.send_message(payload)
            return True
    return False


def send_text_message(payload: dict[str, Any]) -> None:
    _emit_text_message(payload)


def _text_info(message: str) -> None:
    with _TEXT_FORWARDER_LOCK:
        if _TEXT_FORWARDER is not None:
            _TEXT_FORWARDER.send_info(message)


def _text_verbose(message: str) -> None:
    with _TEXT_FORWARDER_LOCK:
        if _TEXT_FORWARDER is not None:
            _TEXT_FORWARDER.send_verbose(message)


def _text_warning(message: str) -> None:
    with _TEXT_FORWARDER_LOCK:
        if _TEXT_FORWARDER is not None:
            _TEXT_FORWARDER.send_warning(message)


def _text_error(message: str) -> None:
    with _TEXT_FORWARDER_LOCK:
        if _TEXT_FORWARDER is not None:
            _TEXT_FORWARDER.send_error(message)


def _text_key_value(key: str, value: object) -> None:
    send_text_message(key_value_pair_message(str(key), str(value)))


def _text_worker_progress(worker_name: str, progress: float, label: str) -> None:
    send_text_message(
        worker_progress_message(str(worker_name), float(progress), str(label))
    )


def _text_master_progress(progress: float, label: str) -> None:
    send_text_message(master_progress_message(float(progress), str(label)))


def _text_delete_worker_bar(worker_name: str) -> None:
    send_text_message(delete_worker_bar_message(str(worker_name)))


atexit.register(shutdown_text_forwarder)
