"""Check that the code on disk is the code that was released (standard library only).

``src/t1dretrain/code_manifest.json`` lists a SHA-256 hash of every Python
file of ``src/`` and ``scripts/`` as released in the project zip (written by
``tools/build_code_manifest.py``). Scripts call ``verify_code()`` at start-up:
if some files were not copied to the server, or were copied from an older
zip, the script stops before spending minutes on a run with mixed code, and
lists the files to copy. Line endings are normalised, so a copy through
Windows does not count as a change.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = Path(__file__).resolve().parent / "code_manifest.json"
CODE_DIRS = ("src", "scripts")


def file_hash(path: Path) -> str:
    """SHA-256 of a text file with line endings normalised to LF."""
    data = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def code_files(root: Path = PROJECT_ROOT) -> list[Path]:
    files = []
    for folder in CODE_DIRS:
        files += [p for p in (root / folder).rglob("*.py") if "__pycache__" not in p.parts]
    return sorted(files)


def build_manifest(version: str, root: Path = PROJECT_ROOT) -> dict:
    return {"version": version,
            "files": {p.relative_to(root).as_posix(): file_hash(p) for p in code_files(root)}}


def check_code(root: Path = PROJECT_ROOT) -> tuple[str, list[str], list[str]]:
    """(released version, files that differ from the release, files missing on disk)."""
    if not MANIFEST.exists():
        return "unknown", [], []
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    changed, missing = [], []
    for relative, expected in manifest["files"].items():
        path = root / relative
        if not path.exists():
            missing.append(relative)
        elif file_hash(path) != expected:
            changed.append(relative)
    return manifest.get("version", "unknown"), changed, missing


def verify_code(strict: bool = True) -> str:
    """Stop (or warn, if not strict) when the code on disk does not match the released manifest."""
    version, changed, missing = check_code()
    if version == "unknown":
        print("WARNING: src/t1dretrain/code_manifest.json not found; cannot check that the code is up to date.")
        return version
    if not changed and not missing:
        print(f"Code check: all files match release {version}.")
        return version
    lines = [f"The code on disk does not match release {version} (src/t1dretrain/code_manifest.json)."]
    if changed:
        lines.append("Files that differ (older copy, or edited): " + ", ".join(changed))
    if missing:
        lines.append("Files missing: " + ", ".join(missing))
    lines.append("Unzip the latest t1d-retraining.zip over the project folder (config/paths.yaml and "
                 "config/environment.local.json are not in the zip and stay as they are), or run with "
                 "--skip-code-check if you changed a file on purpose.")
    message = "\n".join(lines)
    if strict:
        print(message, file=sys.stderr)
        raise SystemExit(2)
    print("WARNING: " + message)
    return version
