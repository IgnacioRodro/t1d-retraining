"""Configuration loading and path resolution.

All scripts read their settings from the YAML files in ``config/``:

- ``project.yaml``  global experimental settings (seeds, horizons, warm-up...).
- ``datasets.yaml`` static facts about each dataset (role, sensor, units...).
- ``paths.yaml``    local folders where the raw datasets live. This file is
  machine specific and is created by copying ``paths.example.yaml``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"
RESULTS_DIR = PROJECT_ROOT / "results"
DATA_DIR = PROJECT_ROOT / "data"


def load_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML file and return a dictionary (empty if the file is empty)."""
    with open(path, encoding="utf-8") as handle:
        content = yaml.safe_load(handle)
    return content or {}


def load_project_config() -> dict[str, Any]:
    """Global experimental settings from ``config/project.yaml``."""
    return load_yaml(CONFIG_DIR / "project.yaml")


def load_datasets_config() -> dict[str, Any]:
    """Static dataset facts from ``config/datasets.yaml``."""
    return load_yaml(CONFIG_DIR / "datasets.yaml")["datasets"]


def load_paths(paths_file: Path | None = None) -> dict[str, Path]:
    """Local raw-data folders from ``config/paths.yaml``.

    Returns a mapping ``dataset_key -> absolute Path``. Relative paths are
    resolved against the folder that contains the YAML file (not against the
    current working directory), so ``../data/raw/REPLACE-BG`` written in
    ``config/paths.yaml`` points to ``<project>/data/raw/REPLACE-BG`` wherever
    the script is launched from. ``~`` is expanded. Entries left empty are
    skipped, so a user can work with a subset of the datasets.
    """
    paths_file = Path(paths_file or (CONFIG_DIR / "paths.yaml")).resolve()
    if not paths_file.exists():
        raise FileNotFoundError(
            f"{paths_file} not found. Copy config/paths.example.yaml to "
            "config/paths.yaml and set the folder of each dataset."
        )
    raw = load_yaml(paths_file).get("datasets", {}) or {}
    return {
        key: resolve_relative(value, paths_file.parent)
        for key, value in raw.items()
        if value
    }


def derived_data_dir(paths_file: Path | None = None) -> Path:
    """Folder for derived data (processed tables, features, runs, models).

    Defaults to ``<project>/data``. Set ``storage.derived_data`` in
    ``config/paths.yaml`` to move it to a fast local disk, for instance when the
    project lives in a cloud-synced folder or on a compute server scratch disk.
    """
    paths_file = Path(paths_file or (CONFIG_DIR / "paths.yaml")).resolve()
    if paths_file.exists():
        value = (load_yaml(paths_file).get("storage", {}) or {}).get("derived_data")
        if value:
            return resolve_relative(value, paths_file.parent)
    return DATA_DIR


def resolve_relative(value: str | Path, base_dir: Path) -> Path:
    """Absolute path; a relative ``value`` is taken relative to ``base_dir``."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()
