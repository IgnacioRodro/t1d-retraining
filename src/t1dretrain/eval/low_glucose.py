"""Forecast lows (scripts 70 and 71; HANDOFF 7.10, D95, D96): why the learned models miss them, and what helps.

Everything here is computed from the Clarke histograms that every evaluation
saves (rule R6: counts of forecasts per 5-mg/dL bin of the reference and of
the forecast, with the Clarke zone of each pair) and from the counts of the day
metrics (forecasts whose reference is below 70 and 54 mg/dL, those that forecast
it, and those that do both):

- ``range_table``: per glucose range of the reference (``RANGES``), the
  number of forecasts, the mean forecast and reference, the bias, MAE and RMSE
  (from the bin centres, so within a few tenths of a mg/dL) and the share in
  each Clarke zone: zone D in the low range is a missed low, in the high range a
  missed high;
- ``calibration``: the mean forecast for every reference bin (E[forecast |
  reference]) and the mean reference for every forecast bin (E[reference |
  forecast]). A model trained on the squared error forecasts the conditional
  mean: the second follows the diagonal and the first is pulled towards the
  middle (regression to the mean), most where the future is least certain;
- ``alarm_curves``: for a low of the reference (below 70 mg/dL, and below 55,
  the bin edge next to 54), the sensitivity and the precision of an alarm that
  sounds when the forecast is below each threshold of ``ALARMS`` (both exact on
  5-mg/dL bins). Moving the threshold moves a model along its curve; a model
  is better at detecting lows only when its curve lies above another's;
- ``matched``: per model, the sensitivity at the precision that persistence
  reaches with the alarm at the event threshold (and the threshold it takes),
  the precision at persistence's sensitivity, and the average precision over
  the curve;
- ``patient_lows`` and ``paired_lows``: the sensitivity and precision of every
  patient at the thresholds of the day metrics (alarm at the event threshold),
  and paired differences between models, arms or losses (bootstrap over
  patients).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import ZONES
from .stats import paired_by

BIN = 5.0  # mg/dL, the bins of the Clarke histograms (eval.metrics.ClarkeAccumulator)
RANGES = (("below 55", 0.0, 55.0), ("55-69", 55.0, 70.0), ("70-179", 70.0, 180.0), ("180-249", 180.0, 250.0),
          ("250 and above", 250.0, np.inf))  # of the reference, on bin edges (54 and 180 fall in the next bin up)
EVENTS = {"below70": 70.0, "below55": 55.0}  # lows of the reference for the alarm curves (bin edges)
ALARMS = tuple(float(a) for a in range(40, 155, 5))  # forecast thresholds of the alarm curves (mg/dL)
REFERENCE_MODEL = "persistence"


def pooled(histogram: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """The histogram summed over every column but ``keys`` (e.g. over seeds)."""
    return histogram.groupby(list(keys) + ["ref_bin", "pred_bin", "zone"], as_index=False, observed=True)["count"].sum()


def range_of(ref_bin: pd.Series) -> pd.Series:
    labels = pd.Series(pd.NA, index=ref_bin.index, dtype="object")
    for name, low, high in RANGES:
        labels[(ref_bin >= low) & (ref_bin < high)] = name
    return labels


def range_table(histogram: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Per key and range of the reference: forecasts, their share, mean reference and forecast, bias, MAE and RMSE
    (bin centres) and the share in each Clarke zone (%)."""
    h = pooled(histogram, keys)
    h["range"] = range_of(h["ref_bin"])
    h["r"], h["f"] = h["ref_bin"] + BIN / 2, h["pred_bin"] + BIN / 2
    h["e"] = h["f"] - h["r"]
    for column in ("r", "f", "e"):
        h[f"{column}_n"] = h[column] * h["count"]
    h["ae_n"], h["se_n"] = h["e"].abs() * h["count"], h["e"] ** 2 * h["count"]
    for z in ZONES:
        h[f"zone_{z}"] = np.where(h["zone"] == z, h["count"], 0)
    sums = ["count", "r_n", "f_n", "e_n", "ae_n", "se_n"] + [f"zone_{z}" for z in ZONES]
    t = h.groupby(list(keys) + ["range"], as_index=False, observed=True)[sums].sum()
    total = t.groupby(list(keys))["count"].transform("sum")
    out = t[list(keys) + ["range"]].copy()
    out["n"] = t["count"]
    out["share_pct"] = 100 * t["count"] / total
    out["mean_reference"] = t["r_n"] / t["count"]
    out["mean_forecast"] = t["f_n"] / t["count"]
    out["bias"] = t["e_n"] / t["count"]
    out["mae"] = t["ae_n"] / t["count"]
    out["rmse"] = np.sqrt(t["se_n"] / t["count"])
    for z in ZONES:
        out[f"clarke_{z}_pct"] = 100 * t[f"zone_{z}"] / t["count"]
        out[f"clarke_{z}_of_all_pct"] = 100 * t[f"zone_{z}"] / total
    order = {name: i for i, (name, _, _) in enumerate(RANGES)}
    return out.sort_values(list(keys) + ["range"], key=lambda c: c.map(order) if c.name == "range" else c) \
        .reset_index(drop=True)


def calibration(histogram: pd.DataFrame, keys: list[str], low: float = 40.0, high: float = 400.0) -> pd.DataFrame:
    """Per key and bin from ``low`` to ``high``: the mean forecast given the reference bin and the mean reference
    given the forecast bin (bin centres), with their counts."""
    h = pooled(histogram, keys)
    h = h.groupby(list(keys) + ["ref_bin", "pred_bin"], as_index=False)["count"].sum()
    parts = []
    for side, other, name in (("ref_bin", "pred_bin", "forecast_given_reference"),
                              ("pred_bin", "ref_bin", "reference_given_forecast")):
        g = h.assign(v=(h[other] + BIN / 2) * h["count"]).groupby(list(keys) + [side], as_index=False)[
            ["v", "count"]].sum()
        g = g[(g[side] >= low) & (g[side] < high)]
        parts.append(pd.DataFrame({**{k: g[k] for k in keys}, "kind": name, "bin": g[side] + BIN / 2,
                                   "mean": g["v"] / g["count"], "n": g["count"]}))
    return pd.concat(parts, ignore_index=True)


def alarm_curves(histogram: pd.DataFrame, keys: list[str], events: dict = EVENTS,
                 alarms: tuple = ALARMS) -> pd.DataFrame:
    """Per key, event and alarm threshold: lows of the reference, alarms, hits, sensitivity, precision and F1
    (%), and the false alarms per 100 forecasts."""
    h = pooled(histogram, keys).groupby(list(keys) + ["ref_bin", "pred_bin"], as_index=False)["count"].sum()
    rows = []
    for key, g in h.groupby(list(keys), sort=True):
        key = key if isinstance(key, tuple) else (key,)
        ref, pred, count = g["ref_bin"].to_numpy(), g["pred_bin"].to_numpy(), g["count"].to_numpy(float)
        total = count.sum()
        for event, threshold in events.items():
            low = ref < threshold
            n_low = count[low].sum()
            for alarm in alarms:
                sounding = pred < alarm
                alarms_n, hits = count[sounding].sum(), count[low & sounding].sum()
                sensitivity = 100 * hits / n_low if n_low else np.nan
                precision = 100 * hits / alarms_n if alarms_n else np.nan
                f1 = 2 * sensitivity * precision / (sensitivity + precision) if sensitivity + precision > 0 else np.nan
                rows.append({**dict(zip(keys, key)), "event": event, "event_threshold": threshold, "alarm": alarm,
                             "lows": n_low, "alarms": alarms_n, "hits": hits, "sensitivity_pct": sensitivity,
                             "precision_pct": precision, "f1_pct": f1,
                             "false_alarms_per_100": 100 * (alarms_n - hits) / total if total else np.nan})
    return pd.DataFrame(rows)


def _at(curve: pd.DataFrame, x: str, y: str, value: float) -> tuple[float, float, str]:
    """``y`` and the alarm threshold where ``x`` reaches ``value`` along the curve (linear between thresholds).
    Sensitivity rises with the threshold: its first crossing. Precision mostly falls but not always (few alarms at
    low thresholds): the highest threshold whose precision is still at least ``value``, the most sensitive alarm that
    is that precise. 'beyond' when it is still reached at the highest threshold (a lower bound), 'below' when it is
    never reached."""
    c = curve.sort_values("alarm")
    xs, ys, alarms = c[x].to_numpy(float), c[y].to_numpy(float), c["alarm"].to_numpy(float)
    ok = np.isfinite(xs) & np.isfinite(ys)
    xs, ys, alarms = xs[ok], ys[ok], alarms[ok]
    if not len(xs) or not np.isfinite(value):
        return np.nan, np.nan, ""

    def between(i: int) -> tuple[float, float, str]:
        a, b = xs[i], xs[i + 1]
        w = 0.0 if b == a else (value - a) / (b - a)
        return float(ys[i] + w * (ys[i + 1] - ys[i])), float(alarms[i] + w * (alarms[i + 1] - alarms[i])), ""

    if x == "sensitivity_pct":
        for i in range(len(xs) - 1):
            if xs[i] <= value <= xs[i + 1]:
                return between(i)
        return (np.nan, np.nan, "below") if xs[0] > value else (float(ys[-1]), float(alarms[-1]), "beyond")
    reached = np.flatnonzero(xs >= value)
    if not len(reached):
        return np.nan, np.nan, "below"
    i = int(reached[-1])
    return (float(ys[i]), float(alarms[i]), "beyond") if i == len(xs) - 1 else between(i)


def average_precision(curve: pd.DataFrame) -> float:
    """Sum over the thresholds of (increase in sensitivity) x precision, from sensitivity 0 (%)."""
    c = curve.sort_values("alarm")
    s, p = c["sensitivity_pct"].to_numpy(float), c["precision_pct"].to_numpy(float)
    ok = np.isfinite(s) & np.isfinite(p)
    s, p = s[ok], p[ok]
    if not len(s):
        return np.nan
    return float(np.sum(np.diff(np.r_[0.0, s]) * p) / 100)


def matched(curves: pd.DataFrame, keys: list[str], reference: str = REFERENCE_MODEL,
            ignore: tuple = ()) -> pd.DataFrame:
    """Per key and event: sensitivity and precision with the alarm at the event threshold; the sensitivity (and
    threshold) at the precision that ``reference`` reaches there and the precision at its sensitivity; the average
    precision. ``keys`` must hold ``model``; the reference is looked up with the same values of the other keys
    except those in ``ignore`` (e.g. the arm: persistence has a single one)."""
    others = [k for k in keys if k != "model" and k not in ignore] + ["event"]
    lookup = {}
    for key, g in curves[curves["model"] == reference].groupby(others, sort=False):
        at = g[g["alarm"] == g["event_threshold"]]
        if len(at):
            lookup[key if isinstance(key, tuple) else (key,)] = (float(at["sensitivity_pct"].iloc[0]),
                                                                  float(at["precision_pct"].iloc[0]))
    rows = []
    for key, g in curves.groupby(list(keys) + ["event"], sort=False):
        values = dict(zip(list(keys) + ["event"], key if isinstance(key, tuple) else (key,)))
        at = g[g["alarm"] == g["event_threshold"]]
        sens_ref, prec_ref = lookup.get(tuple(values[k] for k in others), (np.nan, np.nan))
        sens_m, alarm_m, flag_m = _at(g, "precision_pct", "sensitivity_pct", prec_ref)
        prec_m, alarm_s, flag_s = _at(g, "sensitivity_pct", "precision_pct", sens_ref)
        rows.append({**values, "lows": float(g["lows"].iloc[0]),
                     "sensitivity_pct": float(at["sensitivity_pct"].iloc[0]) if len(at) else np.nan,
                     "precision_pct": float(at["precision_pct"].iloc[0]) if len(at) else np.nan,
                     "reference_sensitivity_pct": sens_ref, "reference_precision_pct": prec_ref,
                     "sensitivity_at_reference_precision_pct": sens_m, "alarm_for_reference_precision": alarm_m,
                     "note_precision": flag_m, "precision_at_reference_sensitivity_pct": prec_m,
                     "alarm_for_reference_sensitivity": alarm_s, "note_sensitivity": flag_s,
                     "average_precision_pct": average_precision(g)})
    return pd.DataFrame(rows)


def shrinkage(histogram: pd.DataFrame, keys: list[str], threshold: float = 70.0) -> pd.DataFrame:
    """Per key: the mean reference and forecast when the reference is below ``threshold`` (how far above a true low
    the forecast lies), and the mean reference when the forecast is below it (how deep a forecast low really is)."""
    h = pooled(histogram, keys).groupby(list(keys) + ["ref_bin", "pred_bin"], as_index=False)["count"].sum()
    h["r"], h["f"] = (h["ref_bin"] + BIN / 2) * h["count"], (h["pred_bin"] + BIN / 2) * h["count"]
    rows = []
    for key, g in h.groupby(list(keys), sort=False):
        key = key if isinstance(key, tuple) else (key,)
        low, said = g["ref_bin"] < threshold, g["pred_bin"] < threshold
        n_low, n_said = g.loc[low, "count"].sum(), g.loc[said, "count"].sum()
        rows.append({**dict(zip(keys, key)), "reference_lows": n_low,
                     "mean_reference_in_lows": g.loc[low, "r"].sum() / n_low if n_low else np.nan,
                     "mean_forecast_in_lows": g.loc[low, "f"].sum() / n_low if n_low else np.nan,
                     "forecast_lows": n_said,
                     "mean_reference_of_forecast_lows": g.loc[said, "r"].sum() / n_said if n_said else np.nan})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["forecast_above_true_low"] = out["mean_forecast_in_lows"] - out["mean_reference_in_lows"]
    return out


# ============================================================================ per patient (day metrics)


def thresholds_of(days: pd.DataFrame) -> list[int]:
    return sorted({int(float(c[len("below"):-len("_true")])) for c in days.columns
                   if c.startswith("below") and c.endswith("_true")}, reverse=True)


def patient_lows(days: pd.DataFrame, keys: list[str], min_events: int) -> pd.DataFrame:
    """Per key, patient and threshold of the day metrics: reference lows, alarms (forecast below the threshold),
    hits, sensitivity (patients with at least ``min_events`` lows) and precision (at least ``min_events`` alarms)."""
    rows = []
    for t in thresholds_of(days):
        columns = [f"below{t}_{k}" for k in ("true", "alarm", "hit")]
        g = days.groupby(list(keys) + ["patient_id"], as_index=False)[columns].sum()
        g.columns = list(keys) + ["patient_id", "lows", "alarms", "hits"]
        g["threshold"] = t
        g["sensitivity_pct"] = np.where(g["lows"] >= min_events, 100 * g["hits"] / g["lows"].where(g["lows"] > 0),
                                        np.nan)
        g["precision_pct"] = np.where(g["alarms"] >= min_events,
                                      100 * g["hits"] / g["alarms"].where(g["alarms"] > 0), np.nan)
        rows.append(g)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def paired_lows(patients: pd.DataFrame, keys: list[str], column: str, pairs: dict, replicates: int,
                seed: int) -> pd.DataFrame:
    """Paired per patient: for every value ``a`` of ``column`` with ``pairs[a] = b``, the mean difference a - b of
    the sensitivity and the precision (points, bootstrap interval over patients), within the other ``keys`` and
    the threshold."""
    table = paired_by(patients, list(keys) + ["threshold"], column, pairs, ["sensitivity_pct", "precision_pct"],
                      replicates, seed)
    if not table.empty:
        table["measure"] = table["measure"].str.replace("_pct", "", regex=False)
    return table
