"""Carbohydrates on board (COB) with linear absorption (script 05, D54).

Each entry of g grams at time t is absorbed at a constant rate over
``absorption_min`` minutes (default 180): COB(T) = sum of g x max(0, 1 - (T - t)
/ absorption_min) over past entries, and the absorption rate is g /
absorption_min while the entry is being absorbed. Entries go to the minute bin
ceil(t), as insulin does, so the inputs at T only contain entries made at or
before T.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import oaconvolve


def carbs_per_minute(carbs: pd.DataFrame, origin: pd.Timestamp, n_minutes: int) -> np.ndarray:
    grams = np.zeros(n_minutes)
    if carbs.empty:
        return grams
    t = (carbs["ts"] - origin).dt.total_seconds().to_numpy(float) / 60.0
    g = carbs["carbs_g"].to_numpy(float)
    bins = np.ceil(t).astype(np.int64)
    ok = (bins >= 0) & (bins < n_minutes) & np.isfinite(g) & (g > 0)
    np.add.at(grams, bins[ok], g[ok])
    return grams


def on_board(grams: np.ndarray, absorption_min: float = 180.0) -> tuple[np.ndarray, np.ndarray]:
    """Per-minute COB (g) and absorption rate (g/min)."""
    lags = np.arange(int(np.ceil(absorption_min)) + 1)
    remaining = np.clip(1 - lags / absorption_min, 0, None)
    rate = np.where(lags < absorption_min, 1 / absorption_min, 0.0)
    cob = oaconvolve(grams, remaining)[:len(grams)]
    absorption = oaconvolve(grams, rate)[:len(grams)]
    return np.where(cob < 1e-9, 0.0, cob), np.where(absorption < 1e-12, 0.0, absorption)  # FFT noise -> 0
