from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

VOLUME_NAME = "credit-assignment-modal-service-state"
REMOTE_MOUNT_PATH = "/volume/small_files"
LOCAL_DIRNAME = "volume_small_files"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _volume_path_from_mount_path(mount_path: str) -> str:
    normalized = mount_path.strip()
    if not normalized.startswith("/"):
        normalized = f"/{normalized}"
    if normalized.startswith("/volume/"):
        return normalized[len("/volume") :]
    if normalized == "/volume":
        return "/"
    return normalized


def _reset_local_destination(local_destination: Path) -> None:
    if local_destination.is_dir():
        shutil.rmtree(local_destination)
    elif local_destination.exists():
        local_destination.unlink()


def main() -> int:
    repo_root = _repo_root()
    local_destination = repo_root / LOCAL_DIRNAME
    remote_volume_path = _volume_path_from_mount_path(REMOTE_MOUNT_PATH)

    _reset_local_destination(local_destination)

    command = [
        "uv",
        "run",
        "modal",
        "volume",
        "get",
        "--force",
        VOLUME_NAME,
        remote_volume_path,
        str(local_destination),
    ]
    result = subprocess.run(command, cwd=str(repo_root), check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"modal volume get failed with exit code {result.returncode}"
        )

    print(
        f"Downloaded {REMOTE_MOUNT_PATH} from volume '{VOLUME_NAME}' to {local_destination}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
