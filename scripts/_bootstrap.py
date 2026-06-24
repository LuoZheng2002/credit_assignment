from __future__ import annotations

import os
import sys
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


_PROXY_ENV_VARS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)


def _normalize_proxy_url(value: str) -> str:
    parsed = urlsplit(value)
    if parsed.scheme == "socks":
        return urlunsplit(parsed._replace(scheme="socks5"))
    return value


for _proxy_env_var in _PROXY_ENV_VARS:
    _proxy_value = os.environ.get(_proxy_env_var)
    if _proxy_value:
        os.environ[_proxy_env_var] = _normalize_proxy_url(_proxy_value)
