"""Two-phase, fully parallel harmonisation of the raw datasets (script 02).

Phase A - read. Each dataset loader (``t1dretrain.io.<dataset>``) splits its
work into independent tasks: one big file, one chunked file, one patient's
folder, one batch of daily wearable files... Tasks from every dataset run
together on all workers (largest first). Each task writes the rows it reads as
per-patient parquet parts and returns a ``TaskReport`` (rows read, value counts
of the source categories it mapped, per-patient metadata, device periods).

Phase B - merge. One job per patient concatenates its parts table by table,
applies the dataset's ``finalize`` step (e.g. basal-rate durations that need
the patient's whole sorted series), conforms the tables to the common schema
(``t1dretrain.data.schema``), writes them to the processed store and returns a
descriptive summary (``t1dretrain.data.summary``).

A loader module provides:

    DATASET               dataset key (as in config/datasets.yaml)
    plan(root, options)   -> list[Task]; one task may set ``report.payload`` to the
                             static patients table
    finalize(table, frame, meta, aux) -> frame  (optional; runs before ``conform``, so it
                             can use extra columns written in phase A)
    AUX_TABLES            names of auxiliary per-patient part tables, passed to
                             ``finalize`` and ``describe_aux`` as ``aux`` (optional)
    describe_aux(aux, meta) -> dict             (optional per-patient check row)
"""

from __future__ import annotations

import importlib
import shutil
import time
import traceback
import warnings
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..io.common import PartWriter, TaskReport
from .schema import TABLES, conform
from .store import ProcessedStore
from .summary import PatientDescription, describe_patient

DEFAULT_BASAL_RATE_CAP_MIN = 720.0  # a rate-change log carries a rate forward for at most 12 h


@dataclass
class Task:
    dataset: str
    name: str
    function: Callable[[dict, PartWriter, TaskReport, dict], None]
    args: dict = field(default_factory=dict)
    cost: float = 1.0  # relative size, used to start the largest tasks first


@dataclass
class TaskResult:
    dataset: str
    name: str
    status: str
    seconds: float
    rows_read: int
    rows_written: dict
    report: TaskReport | None
    error: str = ""
    warnings: str = ""  # Python/numpy warnings raised inside the task, with the project line that triggered them


class WarningRecorder:
    """Collects warnings (e.g. numpy overflow) with the project code line that caused them.

    Workers print a warning only once and without context; recording it here
    puts it in task_log.csv next to the task, with the t1dretrain call chain.
    """

    def __init__(self) -> None:
        self.counts: Counter = Counter()

    def __enter__(self) -> "WarningRecorder":
        self._context = warnings.catch_warnings()
        self._context.__enter__()
        warnings.simplefilter("always")
        for category in (DeprecationWarning, PendingDeprecationWarning, FutureWarning):
            warnings.filterwarnings("ignore", category=category)  # library notices, not data problems
        warnings.showwarning = self._record
        return self

    def __exit__(self, *exc) -> None:
        self._context.__exit__(*exc)

    def _record(self, message, category, filename, lineno, file=None, line=None) -> None:
        frames = [f for f in traceback.extract_stack()
                  if "t1dretrain" in f.filename and f.name not in {"execute_task", "merge_patient", "_record"}]
        chain = " < ".join(f"{Path(f.filename).name}:{f.lineno} {f.name}" for f in reversed(frames[-3:]))
        self.counts[f"{category.__name__}: {str(message)[:120]} [{Path(filename).name}:{lineno}] from {chain}"] += 1

    def text(self) -> str:
        return " | ".join(f"{n}x {key}" for key, n in self.counts.most_common(10))


def parts_dir(interim_root: Path, dataset: str) -> Path:
    return interim_root / "02_parts" / dataset


def execute_task(job: tuple[Task, Path, dict]) -> TaskResult:
    """Run one phase-A task (in a worker process)."""
    task, parts_root, options = job
    start = time.time()
    writer = PartWriter(parts_root, task.name, flush_rows=options.get("flush_rows", 3_000_000))
    report = TaskReport()
    with WarningRecorder() as recorder:
        try:
            task.function(task.args, writer, report, options)
            writer.close()
            status, error = "ok", ""
        except Exception:  # noqa: BLE001 - a failed task is reported, the run continues
            writer.close()
            status, error = "error", traceback.format_exc(limit=6)
    return TaskResult(task.dataset, task.name, status, round(time.time() - start, 2), report.rows_read,
                      dict(writer.rows_written), report, error, recorder.text())


def merge_meta(results: list[TaskResult]) -> dict[str, dict[str, dict[str, Any]]]:
    """dataset -> patient -> key -> value; keys ending in '_n' are summed, others joined if they differ."""
    merged: dict[str, dict[str, dict[str, Any]]] = {}
    for result in results:
        if result.report is None:
            continue
        per_dataset = merged.setdefault(result.dataset, {})
        for patient_id, values in result.report.meta.items():
            target = per_dataset.setdefault(patient_id, {})
            for key, value in values.items():
                if key.endswith("_n"):
                    target[key] = target.get(key, 0) + value
                elif key not in target or target[key] == value:
                    target[key] = value
                else:
                    known = str(target[key]).split(" + ")
                    if str(value) not in known:
                        target[key] = " + ".join(known + [str(value)])
    return merged


def fill_basal_rate_durations(frame: pd.DataFrame, cap_min: float) -> pd.DataFrame:
    """Durations of basal-rate rows that lack one: time to the next rate row, capped at ``cap_min``.

    Rate-change logs (DCLP3 pumps, T1D-UOM) give only the moment a rate starts;
    the rate holds until the next change. Rows with a recorded duration are
    left as they are.
    """
    if frame.empty or not (frame["kind"] == "basal_rate").any():
        return frame
    frame = frame.copy()
    for column in TABLES["insulin"]:
        if column not in frame:
            frame[column] = np.nan
    frame["ts"] = pd.to_datetime(frame["ts"])
    frame["duration_min"] = pd.to_numeric(frame["duration_min"], errors="coerce")
    rates = frame[frame["kind"] == "basal_rate"].sort_values("ts", kind="stable")
    others = frame[frame["kind"] != "basal_rate"]
    missing = rates["duration_min"].isna()
    if not missing.any():
        return frame
    known, unknown = rates[~missing], rates[missing].drop_duplicates(subset=["ts"], keep="last")
    starts = pd.concat([known["ts"], unknown["ts"]]).drop_duplicates().sort_values()
    next_start = pd.Series(starts.shift(-1).to_numpy(), index=starts.to_numpy())
    gap = (next_start.reindex(unknown["ts"].to_numpy()).to_numpy() - unknown["ts"].to_numpy())
    gap_min = pd.to_timedelta(gap).total_seconds() / 60
    unknown = unknown.assign(duration_min=np.minimum(gap_min, cap_min))
    return pd.concat([others, known, unknown], ignore_index=True)


@dataclass
class MergeResult:
    dataset: str
    patient_id: str
    rows: dict
    description: PatientDescription | None
    seconds: float
    error: str = ""
    counts: dict = field(default_factory=dict)   # (table, column, value) -> n, e.g. how local times were obtained
    extra: dict = field(default_factory=dict)    # loader-specific per-patient checks (``describe_aux``)
    warnings: str = ""


def _read_parts(folder: Path, columns: list[str]) -> pd.DataFrame:
    parts = sorted(folder.glob("*.parquet")) if folder.exists() else []
    return pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True) if parts else pd.DataFrame(
        columns=columns)


def merge_patient(job: dict) -> MergeResult:
    """Phase B for one patient (in a worker process)."""
    with WarningRecorder() as recorder:
        result = _merge_patient(job)
    result.warnings = recorder.text()
    return result


def _merge_patient(job: dict) -> MergeResult:
    start = time.time()
    dataset, patient_id = job["dataset"], job["patient_id"]
    try:
        module = importlib.import_module(job["module"])
        finalize = getattr(module, "finalize", None)
        cap = float(job["meta"].get("basal_rate_cap_min", getattr(module, "BASAL_RATE_CAP_MIN",
                                                                    DEFAULT_BASAL_RATE_CAP_MIN)))
        store = ProcessedStore(dataset, root=job["derived_root"])
        aux = {name: _read_parts(job["parts_root"] / name / patient_id, [])
               for name in getattr(module, "AUX_TABLES", [])}
        tables: dict[str, pd.DataFrame] = {}
        rows: dict[str, int] = {}
        counts: dict = {}
        for table in TABLES:
            frame = _read_parts(job["parts_root"] / table / patient_id, list(TABLES[table]))
            frame["patient_id"] = patient_id
            if finalize is not None and not frame.empty:
                frame = finalize(table, frame, job["meta"], aux)  # may use loader-specific extra columns
            if "time_origin" in frame:
                for value, n in frame["time_origin"].value_counts().items():
                    counts[(table, "local time from (final)", str(value))] = int(n)
            if table == "insulin" and not frame.empty:
                frame = fill_basal_rate_durations(frame, cap)
            frame = conform(table, frame)
            rows[table] = store.write(table, patient_id, frame) if not frame.empty else 0
            tables[table] = frame
        describe_aux = getattr(module, "describe_aux", None)
        extra = describe_aux(aux, job["meta"]) if describe_aux is not None else {}
        meta = job["meta"]
        window = (meta["window_start"], meta["window_end"]) if meta.get("window_start") and meta.get("window_end") \
            else None
        description = describe_patient(dataset, patient_id, tables, job["coverage_threshold"], window)
        clock = {value: n for (table, column, value), n in counts.items() if table == "cgm"}
        for value, n in clock.items():  # how the local times of this patient's CGM rows were obtained (Loop)
            description.row[f"cgm_time_{value}_share"] = round(n / sum(clock.values()), 4)
        return MergeResult(dataset, patient_id, rows, description, round(time.time() - start, 2), counts=counts,
                           extra=extra)
    except Exception:  # noqa: BLE001
        return MergeResult(dataset, patient_id, {}, None, round(time.time() - start, 2),
                           traceback.format_exc(limit=6))


def list_patients(parts_root: Path) -> list[str]:
    """Patients that have at least one part file, in any table."""
    patients: set[str] = set()
    for table in TABLES:
        folder = parts_root / table
        if folder.exists():
            patients.update(p.name for p in folder.iterdir() if p.is_dir())
    return sorted(patients)


def reset_dataset(derived_root: Path, interim_root: Path, dataset: str) -> None:
    """Remove the previous outputs of one dataset (processed tables and parts)."""
    ProcessedStore(dataset, root=derived_root).clear()
    parts = parts_dir(interim_root, dataset)
    if parts.exists():
        shutil.rmtree(parts)
    parts.mkdir(parents=True, exist_ok=True)


def value_counts_frame(results: list[TaskResult]) -> pd.DataFrame:
    counts: Counter = Counter()
    for result in results:
        if result.report is None:
            continue
        for (source, column, value), n in result.report.value_counts.items():
            counts[(result.dataset, source, column, value)] += n
    rows = [{"dataset": d, "source": s, "column": c, "value": v, "count": n} for (d, s, c, v), n in counts.items()]
    frame = pd.DataFrame(rows, columns=["dataset", "source", "column", "value", "count"])
    return frame.sort_values(["dataset", "source", "column", "count"], ascending=[True, True, True, False])
