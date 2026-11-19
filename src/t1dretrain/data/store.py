"""Per-patient parquet store: ``<derived_data>/<stage>/<dataset>/<table>/<patient_id>.parquet``.

Stages: ``processed`` (script 02, harmonised raw data) and ``clean`` (script 03,
one CGM series per patient after de-duplication and source choice).

One file per patient and table keeps every later step simple and parallel:
a worker reads exactly the patient it needs. ``patients.parquet`` holds one row
per patient with the static information.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import pandas as pd

from ..config import derived_data_dir
from .schema import PATIENT_COLUMNS, TABLES, conform


def safe_id(patient_id: str) -> str:
    """File-name friendly patient id."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(patient_id))


class ProcessedStore:
    """Read and write the harmonised tables of one dataset."""

    def __init__(self, dataset: str, root: Path | None = None, stage: str = "processed") -> None:
        """``stage``: "processed" (script 02, harmonised) or "clean" (script 03, quality-checked)."""
        self.dataset = dataset
        self.root = (root or derived_data_dir()) / stage / dataset

    def clear(self) -> None:
        """Remove previous outputs of this dataset (a fresh run)."""
        if self.root.exists():
            shutil.rmtree(self.root)
        self.root.mkdir(parents=True, exist_ok=True)

    def table_dir(self, table: str) -> Path:
        return self.root / table

    def write(self, table: str, patient_id: str, frame: pd.DataFrame) -> int:
        """Conform and write one patient's table; returns the number of rows written."""
        frame = conform(table, frame.assign(patient_id=str(patient_id)))
        if frame.empty:
            return 0
        target = self.table_dir(table) / f"{safe_id(patient_id)}.parquet"
        target.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(target, index=False)
        return len(frame)

    def write_frame(self, table: str, patient_id: str, frame: pd.DataFrame) -> int:
        """Write one patient's table as given (tables of later stages with their own columns)."""
        if frame.empty:
            return 0
        target = self.table_dir(table) / f"{safe_id(patient_id)}.parquet"
        target.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(target, index=False)
        return len(frame)

    def read_frame(self, table: str, patient_id: str) -> pd.DataFrame:
        target = self.table_dir(table) / f"{safe_id(patient_id)}.parquet"
        return pd.read_parquet(target) if target.exists() else pd.DataFrame()

    def write_grouped(self, table: str, frame: pd.DataFrame) -> dict[str, int]:
        """Split a multi-patient frame by ``patient_id`` and write each part."""
        return {pid: self.write(table, pid, part) for pid, part in frame.groupby("patient_id", sort=False)}

    def read(self, table: str, patient_id: str) -> pd.DataFrame:
        """One patient's table (empty frame if the patient has none)."""
        target = self.table_dir(table) / f"{safe_id(patient_id)}.parquet"
        if not target.exists():
            return conform(table, pd.DataFrame(columns=list(TABLES[table])))
        return pd.read_parquet(target)

    def patients_with(self, table: str) -> list[str]:
        """Patient ids that have a file for ``table``."""
        folder = self.table_dir(table)
        return sorted(p.stem for p in folder.glob("*.parquet")) if folder.exists() else []

    def write_patients(self, patients: pd.DataFrame) -> None:
        """Static patient table (one row per patient)."""
        patients = patients.copy()
        for column in PATIENT_COLUMNS:
            if column not in patients:
                patients[column] = pd.NA
        ordered = PATIENT_COLUMNS + [c for c in patients.columns if c not in PATIENT_COLUMNS]
        patients = patients[ordered].astype({"patient_id": "string", "dataset": "string"})
        for column in ["window_start", "window_end"]:
            patients[column] = pd.to_datetime(patients[column], errors="coerce")
        for column in patients.columns[patients.dtypes == object]:
            kinds = {type(v) for v in patients[column].dropna()}
            if len(kinds) > 1:  # mixed values (e.g. numbers and text from different tasks): store as text
                patients[column] = patients[column].astype("string")
        self.root.mkdir(parents=True, exist_ok=True)
        patients.to_parquet(self.root / "patients.parquet", index=False)

    def read_patients(self) -> pd.DataFrame:
        return pd.read_parquet(self.root / "patients.parquet")

    def write_extra(self, name: str, frame: pd.DataFrame) -> None:
        """Dataset-level side table, e.g. ``device_periods``."""
        self.root.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(self.root / f"{name}.parquet", index=False)

    def read_extra(self, name: str) -> pd.DataFrame:
        target = self.root / f"{name}.parquet"
        return pd.read_parquet(target) if target.exists() else pd.DataFrame()
