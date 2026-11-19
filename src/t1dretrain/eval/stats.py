"""Statistics shared by the analyses: bootstrap over patients (HANDOFF section 7.8)."""

from __future__ import annotations

import numpy as np


def bootstrap_mean(values: np.ndarray, replicates: int = 2000, seed: int = 20261026,
                   level: float = 0.95) -> tuple[float, float, float]:
    """Mean of per-patient values and its percentile bootstrap interval (patients resampled)."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, x.size, size=(replicates, x.size))].mean(axis=1)
    alpha = (1 - level) / 2
    return float(x.mean()), float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha))


def paired_by(table, keys: list[str], column: str, pairs: dict, measures: list[str], replicates: int = 2000,
              seed: int = 20261026):
    """Paired per patient (``table``: one row per patient and value of ``keys``): for every value ``a`` of ``column``
    with ``pairs[a] = b``, the mean difference a - b of each measure with its bootstrap interval over patients,
    within the other ``keys``; at least 3 patients with both."""
    import pandas as pd

    rows = []
    group = [k for k in keys if k != column]
    for key, g in table.groupby(group, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        for measure in measures:
            wide = g.pivot_table(index="patient_id", columns=column, values=measure)
            for a, b in pairs.items():
                if a not in wide or b not in wide or a == b:
                    continue
                d = (wide[a] - wide[b]).dropna().to_numpy()
                if len(d) < 3:
                    continue
                mean, low, high = bootstrap_mean(d, replicates, seed)
                rows.append({**dict(zip(group, key)), column: a, "against": b, "measure": measure,
                             "patients": int(len(d)), "mean_against": float(wide.loc[wide[a].notna(), b].mean()),
                             "difference": mean, "low": low, "high": high, "patients_higher": int((d > 0).sum())})
    return pd.DataFrame(rows)
