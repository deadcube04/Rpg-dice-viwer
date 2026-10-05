"""Immutable identifiers for data and source used by experiment runs."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path


def source_digest(root: Path = Path(".")) -> str:
    digest = hashlib.sha256()
    for path in sorted((root / "src").rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    for name in ("pyproject.toml", "uv.lock", "dvc.yaml"):
        path = root / name
        if path.is_file():
            digest.update(name.encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def git_revision(root: Path = Path(".")) -> str | None:
    result = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None
