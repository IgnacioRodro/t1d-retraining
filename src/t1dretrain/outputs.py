"""Uniform handling of script outputs.

Every script writes its results to ``results/<script_id>/`` (optionally with a
run tag sub-folder). The folder always contains:

- CSV tables with the numerical results,
- PNG figures,
- ``summary.md`` with a short human-readable digest,
- ``run_manifest.json`` with what was run, when, with which settings and which
  files were produced.

The folder is meant to be zipped and shared for review.
"""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless backend: scripts never open windows
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from .config import PROJECT_ROOT, RESULTS_DIR  # noqa: E402


def room_for_title(fig: plt.Figure) -> None:
    """Move the axes down when the figure title overlaps them: ``tight_layout`` ignores ``suptitle``, so in tall
    figures the title fell on the titles of the first row. Never fails the saving of a figure."""
    title = getattr(fig, "_suptitle", None)
    if title is None or not title.get_text() or not fig.axes:
        return
    try:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        title_bottom = title.get_window_extent(renderer).y0
        axes_top = max(ax.get_tightbbox(renderer).y1 for ax in fig.axes if ax.get_visible())
        overlap = axes_top - (title_bottom - 6)  # 6 pixels between the title and the highest axes text
        if overlap > 0:
            fig.subplots_adjust(top=fig.subplotpars.top - overlap / fig.bbox.height)
    except Exception:  # noqa: BLE001
        pass


class ScriptOutput:
    """Collects every artefact a script produces and documents the run.

    ``resources`` (from ``t1dretrain.compute.detect_resources``) is stored in the
    manifest so every result says on which machine, with how many workers and on
    which device it was produced (rule R5). The folder starts empty (``fresh``),
    so it holds only this run's files; script 00 keeps the files of its first
    phase.
    """

    def __init__(self, script_id: str, run_tag: str | None = None, resources: Any = None, fresh: bool = True) -> None:
        self.script_id = script_id
        self.dir = RESULTS_DIR / script_id / run_tag if run_tag else RESULTS_DIR / script_id
        if fresh and self.dir.exists():  # only this run's files: nothing left from an earlier run or release
            shutil.rmtree(self.dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.resources = resources
        self._start = time.time()
        self._files: list[str] = []

    def path(self, name: str) -> Path:
        """Path inside the output folder (sub-folders are created on demand)."""
        target = self.dir / name
        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    def csv(self, frame: pd.DataFrame, name: str) -> Path:
        """Write a table as CSV (comma separated, UTF-8, no index)."""
        target = self.path(name if name.endswith(".csv") else f"{name}.csv")
        frame.to_csv(target, index=False)
        self._register(target)
        return target

    def figure(self, fig: plt.Figure, name: str, dpi: int = 150, formats: tuple[str, ...] = ("png",),
               exact_size: bool = False) -> Path:
        """Save a matplotlib figure in each of ``formats`` (PNG by default; PDF is vector) and close it; returns the
        first file. By default the axes are first moved below the figure title and the white margin is cut; with
        ``exact_size`` the figure keeps its size, as a paper figure must (script 90)."""
        stem = name[:-4] if name.endswith(".png") else name
        if not exact_size:
            room_for_title(fig)
        written = []
        for extension in formats:
            target = self.path(f"{stem}.{extension}")
            fig.savefig(target, dpi=dpi, bbox_inches=None if exact_size else "tight")
            self._register(target)
            written.append(target)
        plt.close(fig)
        return written[0]

    def text(self, content: str, name: str) -> Path:
        """Write a text or markdown file."""
        target = self.path(name)
        target.write_text(content, encoding="utf-8")
        self._register(target)
        return target

    def register(self, name: str) -> None:
        """Add a file written by other means (for example an earlier phase) to the manifest."""
        target = self.dir / name
        if target.exists():
            self._register(target)

    def manifest(self, settings: dict[str, Any] | None = None, notes: str = "") -> Path:
        """Write ``run_manifest.json``; call it last so the file list is complete."""
        manifest = {
            "script_id": self.script_id,
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "duration_seconds": round(time.time() - self._start, 1),
            "command": " ".join(sys.argv),
            "python": sys.version.split()[0],
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "git_commit": _git_commit(),
            "package_version": _package_version(),
            "code_check": _code_check(),
            "resources": self.resources.as_dict() if hasattr(self.resources, "as_dict") else self.resources,
            "settings": settings or {},
            "notes": notes,
            "files": sorted(self._files),
        }
        target = self.path("run_manifest.json")
        target.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
        return target

    def _register(self, target: Path) -> None:
        self._files.append(str(target.relative_to(self.dir)))


def markdown_table(frame: pd.DataFrame, floatfmt: str = "{:,.3g}") -> str:
    """GitHub-style markdown table without extra dependencies."""
    if frame.empty:
        return "(empty)"

    def cell(value) -> str:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return ""
        if isinstance(value, float):
            return f"{value:,.0f}" if value.is_integer() and abs(value) >= 1 else floatfmt.format(value)
        if isinstance(value, int):
            return f"{value:,}"
        return str(value).replace("|", "/").replace("\n", " ")

    header = "| " + " | ".join(str(c) for c in frame.columns) + " |"
    rule = "|" + "|".join("---" for _ in frame.columns) + "|"
    rows = ["| " + " | ".join(cell(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([header, rule] + rows)


def _package_version() -> str:
    from . import __version__

    return __version__


def _code_check() -> dict:
    """Files that differ from the released code manifest (empty lists = code as released)."""
    from .codecheck import check_code

    version, changed, missing = check_code()
    return {"release": version, "changed": changed, "missing": missing}


def _git_commit() -> str | None:
    """Current git commit of the project, if the project is a git repository."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None
