"""Result cache of the simulation scripts (30, 40): one pickle per model, policy or arm, and patient-anchor.

An entry is ``{"path": ..., "key": ..., "recompute": None or the start time of a run with --recompute}``. A cached
result is valid when its key matches (the key covers the settings, the starting point and the simulator's
``CACHE_VERSION``), it holds no fit that failed for a reason other than the data (D72), and, with --recompute, it
was written by this run (the run's start time is stored with the result; no file clock is involved).
"""

from __future__ import annotations

import pickle
from pathlib import Path

DATA_ERRORS = ("ValueError: no training samples", "ValueError: too few training samples")


def retryable(row: dict) -> bool:
    """A fit that failed for a reason other than the data (e.g. CUDA): computed again in the next run."""
    error = row.get("error")
    return isinstance(error, str) and bool(error) and not error.startswith(DATA_ERRORS)


def cache_valid(entry: dict) -> bool:
    path = Path(entry["path"])
    if not path.exists():
        return False
    try:
        with open(path, "rb") as handle:
            content = pickle.load(handle)
    except Exception:  # noqa: BLE001
        return False
    if entry.get("recompute") and content.get("run") != entry["recompute"]:  # --recompute: only this run's results
        return False
    return content.get("key") == entry["key"] and not any(retryable(row) for row in content.get("info", []))


def write_cache(entry: dict, content: dict) -> None:
    """Write through a temporary file of its own, then rename it over the result: several jobs of one patient may
    write the same entry at once (the persistence of scripts 50 and 71-73), and each rename is atomic."""
    import uuid

    path = Path(entry["path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.stem}.{uuid.uuid4().hex}.tmp")
    try:
        with open(temporary, "wb") as handle:
            pickle.dump({"key": entry["key"], "run": entry.get("recompute"), **content}, handle)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def read_cache(entry: dict) -> dict | None:
    if not cache_valid(entry):
        return None
    with open(entry["path"], "rb") as handle:
        return pickle.load(handle)
