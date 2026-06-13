#!/usr/bin/env python3
from pathlib import Path


def update_file(path: Path) -> bool:
    old = '    --mount-dir "/volume" \\\n    --ui true\n'
    new = (
        '    --mount-dir "/volume" \\\n    --keep-action-logs true \\\n    --ui true\n'
    )
    text = path.read_text()
    if "--keep-action-logs true" in text:
        return False
    if old not in text:
        raise RuntimeError(f"expected pattern not found in {path}")
    path.write_text(text.replace(old, new, 1))
    return True


def main() -> None:
    root = Path(__file__).resolve().parent
    targets = [root / "orchestrator", root / "orchestrator_modal"]
    updated = []
    for directory in targets:
        for path in sorted(directory.glob("*.sh")):
            if update_file(path):
                updated.append(path)
    print(f"updated {len(updated)} files")
    for path in updated:
        print(path.relative_to(root.parent))


if __name__ == "__main__":
    main()
