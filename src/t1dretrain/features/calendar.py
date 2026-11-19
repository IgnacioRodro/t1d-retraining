"""Calendar inputs (script 05): time of day always; day of week only where dates are real (D24)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def real_dates(dates_note: object) -> bool:
    """True when the ``dates`` note of the patients table says the dates are real (HUPA-UCM, BrisT1D, T1D-UOM)."""
    return isinstance(dates_note, str) and dates_note.strip().lower().startswith("real")


def time_of_day(ts: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    hours = (ts - ts.dt.normalize()).dt.total_seconds().to_numpy() / 3600.0
    return np.sin(2 * np.pi * hours / 24), np.cos(2 * np.pi * hours / 24)
