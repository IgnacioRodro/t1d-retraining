"""Insulin on board (IOB) and insulin activity from delivered insulin (script 05, D54).

Every insulin record of the common schema becomes delivered units on a
1-minute axis (``delivery_per_minute``):

- a ``bolus`` is given at once;
- a long-acting ``basal_injection`` acts for about a day, so its units are
  spread evenly over ``long_acting_spread_h`` hours (default 24);
- ``bolus_extended`` and ``basal_rate`` spread their units over their
  duration; a basal rate ends early when the next rate starts (the pump log
  is a list of rate changes, so overlapping segments are not added up);
- ``basal_5min`` and ``total_5min`` spread their units over the 5 minutes
  ending at their time stamp.

An event at time t is put in the minute bin ceil(t), and a value read at time
T uses bins <= floor(T): the inputs at T only ever contain insulin given at or
before T.

IOB(T) = sum over past deliveries of units x remaining fraction(T - t) and
activity(T) = sum of units x activity(T - t), with the exponential insulin
model used by Loop and OpenAPS (LoopKit ``ExponentialInsulinModel``): action
duration ``td`` (default 360 min), peak activity ``tp`` (75 min) and an
optional delay. With tau = tp (1 - tp/td) / (1 - 2 tp/td), a = 2 tau/td and
S = 1 / (1 - a + (1 + a) exp(-td/tau)):

    remaining(t) = 1 - S (1 - a) ((t^2 / (tau td (1 - a)) - t/tau - 1) exp(-t/tau) + 1)
    activity(t)  = S / tau^2 * t (1 - t/td) exp(-t/tau)        (fraction per minute)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.signal import oaconvolve


@dataclass(frozen=True)
class ExponentialInsulin:
    action_min: float = 360.0
    peak_min: float = 75.0
    delay_min: float = 0.0

    def _constants(self) -> tuple[float, float, float]:
        td, tp = self.action_min, self.peak_min
        tau = tp * (1 - tp / td) / (1 - 2 * tp / td)
        a = 2 * tau / td
        s = 1 / (1 - a + (1 + a) * np.exp(-td / tau))
        return tau, a, s

    def remaining(self, minutes: np.ndarray) -> np.ndarray:
        """Fraction of a dose still to act ``minutes`` after it was given."""
        tau, a, s = self._constants()
        td = self.action_min
        t = np.asarray(minutes, dtype=float) - self.delay_min
        inside = np.clip(t, 0, td)
        value = 1 - s * (1 - a) * ((inside ** 2 / (tau * td * (1 - a)) - inside / tau - 1) * np.exp(-inside / tau) + 1)
        return np.where(t <= 0, 1.0, np.where(t >= td, 0.0, value))

    def activity(self, minutes: np.ndarray) -> np.ndarray:
        """Fraction of a dose acting per minute, ``minutes`` after it was given."""
        tau, _, s = self._constants()
        td = self.action_min
        t = np.asarray(minutes, dtype=float) - self.delay_min
        value = s / tau ** 2 * t * (1 - t / td) * np.exp(-t / tau)
        return np.where((t <= 0) | (t >= td), 0.0, value)

    @property
    def length_min(self) -> int:
        return int(np.ceil(self.action_min + self.delay_min)) + 1


def _minute(ts: pd.Series, origin: pd.Timestamp) -> np.ndarray:
    return (ts - origin).dt.total_seconds().to_numpy(float) / 60.0


def _spread(series: np.ndarray, start_min: np.ndarray, end_min: np.ndarray, units: np.ndarray) -> None:
    """Add ``units`` evenly over the bins [ceil(start), ceil(end)) of a per-minute series (in place)."""
    n = len(series)
    first = np.ceil(start_min).astype(np.int64)
    last = np.maximum(np.ceil(end_min).astype(np.int64), first + 1)
    ok = np.isfinite(units) & (units > 0) & (last > 0) & (first < n)
    first, last, units = first[ok], last[ok], units[ok]
    per_minute = units / (last - first)
    edges = np.zeros(n + 1)
    np.add.at(edges, np.clip(first, 0, n), per_minute)
    np.add.at(edges, np.clip(last, 0, n), -per_minute)
    series += np.cumsum(edges)[:n]


def delivery_per_minute(insulin: pd.DataFrame, origin: pd.Timestamp, n_minutes: int,
                        long_acting_spread_h: float = 24.0) -> tuple[np.ndarray, np.ndarray]:
    """Delivered units per minute bin since ``origin``: (bolus-like, basal-like)."""
    bolus = np.zeros(n_minutes)
    basal = np.zeros(n_minutes)
    if insulin.empty:
        return bolus, basal
    t = _minute(insulin["ts"], origin)
    kind = insulin["kind"].astype(str).to_numpy()
    units = insulin["units"].to_numpy(float)
    rate = insulin["rate_u_per_h"].to_numpy(float)
    duration = insulin["duration_min"].to_numpy(float)

    instant = kind == "bolus"
    bins = np.ceil(t[instant]).astype(np.int64)
    ok = (bins >= 0) & (bins < n_minutes) & np.isfinite(units[instant]) & (units[instant] > 0)
    np.add.at(bolus, bins[ok], units[instant][ok])

    extended = kind == "bolus_extended"
    _spread(bolus, t[extended], t[extended] + np.nan_to_num(duration[extended], nan=0.0), units[extended])

    segments = np.flatnonzero(kind == "basal_rate")
    if segments.size:
        order = segments[np.argsort(t[segments], kind="stable")]
        start = t[order]
        end = start + np.nan_to_num(duration[order], nan=0.0)
        end = np.minimum(end, np.concatenate([start[1:], [np.inf]]))  # a new rate replaces the running one
        minutes = np.clip(end - start, 0, None)
        _spread(basal, start, end, rate[order] * minutes / 60.0)

    injections = kind == "basal_injection"
    _spread(basal, t[injections], t[injections] + long_acting_spread_h * 60.0, units[injections])

    five = kind == "basal_5min"
    _spread(basal, t[five] - 5.0, t[five], units[five])
    total = kind == "total_5min"
    _spread(basal, t[total] - 5.0, t[total], units[total])  # BrisT1D: bolus and basal together
    return bolus, basal


def on_board(delivery: np.ndarray, model: ExponentialInsulin) -> tuple[np.ndarray, np.ndarray]:
    """Per-minute IOB (U) and activity (U/min) for a per-minute delivery series (causal convolution)."""
    lags = np.arange(model.length_min)
    iob = oaconvolve(delivery, model.remaining(lags))[:len(delivery)]
    activity = oaconvolve(delivery, model.activity(lags))[:len(delivery)]
    return _tidy(iob), _tidy(activity)


def _tidy(values: np.ndarray, floor: float = 1e-9) -> np.ndarray:
    """FFT convolution leaves ~1e-15 noise where nothing was given: set it to exactly 0."""
    return np.where(values < floor, 0.0, values)
