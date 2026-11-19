"""Reference models of stage 0 (HANDOFF section 6.1: M00, M01, M04).

- ``Persistence`` (M00): the glucose at the origin, at every horizon.
- ``LinearExtrapolation`` (M01): straight line fitted by least squares to
  the inputs of the last ``window_min`` minutes, extended to each horizon.
- ``AutoRegressive`` (M04, AR(3)): y_t = c + sum_i phi_i y_(t-i), fitted by
  ordinary least squares on every run of p+1 consecutive inputs of the
  training days within one segment, and iterated to each horizon.
"""

from __future__ import annotations

import numpy as np

from ..sim.samples import Samples
from .base import FitLog, ForecastModel, TrainData, clip


class Persistence(ForecastModel):
    name = "persistence"

    def predict(self, samples: Samples) -> np.ndarray:
        last = samples.inputs[:, -1:]
        return clip(np.repeat(last, len(samples.horizons_min), axis=1))


class LinearExtrapolation(ForecastModel):
    name = "linear_extrapolation"

    def __init__(self, window_min: float = 30.0) -> None:
        self.window_min = window_min

    def predict(self, samples: Samples) -> np.ndarray:
        k = int(round(self.window_min / samples.step_min)) + 1
        y = samples.inputs[:, -k:].astype(float)
        t = (np.arange(k) - (k - 1)) * samples.step_min  # minutes, 0 at the origin
        ok = np.isfinite(y)
        n = ok.sum(axis=1)
        tm = np.where(ok, t, 0).sum(axis=1) / np.maximum(n, 1)
        ym = np.where(ok, y, 0).sum(axis=1) / np.maximum(n, 1)
        dt = np.where(ok, t - tm[:, None], 0)
        slope = (dt * np.where(ok, y - ym[:, None], 0)).sum(axis=1) / np.maximum((dt ** 2).sum(axis=1), 1e-9)
        at_origin = ym + slope * (0 - tm)
        horizons = np.asarray(samples.horizons_min, dtype=float)
        return clip(at_origin[:, None] + slope[:, None] * horizons[None, :])


class AutoRegressive(ForecastModel):
    name = "ar3"
    mechanism = "re_estimation"

    def __init__(self, order: int = 3) -> None:
        self.order = order
        self.coef: np.ndarray | None = None
        self.name = f"ar{order}"

    def fit(self, train: TrainData) -> FitLog:
        p = self.order
        y, seg = train.series, train.segment
        rows = np.arange(p, len(y))
        lagged = np.stack([y[rows - i] for i in range(p + 1)], axis=1)  # column 0 = y_t, column i = y_(t-i)
        ok = np.isfinite(lagged).all(axis=1) & (seg[rows] == seg[rows - p])
        if ok.sum() < 10 * (p + 1):
            self.coef = None
            return FitLog(n_samples=int(ok.sum()), details={"fitted": False})
        design = np.column_stack([np.ones(ok.sum()), lagged[ok, 1:]])
        target = lagged[ok, 0]
        if train.series_weight is not None:  # weighted least squares (stage 2, exponential window)
            root = np.sqrt(np.asarray(train.series_weight, dtype=float)[rows][ok])
            design, target = design * root[:, None], target * root
        self.coef, *_ = np.linalg.lstsq(design, target, rcond=None)
        return FitLog(n_samples=int(ok.sum()), details={"fitted": True, "coef": self.coef.round(4).tolist()})

    def predict(self, samples: Samples) -> np.ndarray:
        if self.coef is None:
            return Persistence().predict(samples)
        p = self.order
        history = samples.inputs[:, -p:].astype(float)
        history = np.where(np.isfinite(history), history, samples.inputs[:, -1:])  # origin is always real
        steps = [int(round(h / samples.step_min)) for h in samples.horizons_min]
        out = np.full((samples.n, len(steps)), np.nan)
        for step in range(1, max(steps) + 1):
            nxt = self.coef[0] + history[:, ::-1] @ self.coef[1:]
            history = np.column_stack([history[:, 1:], nxt])
            for k, s in enumerate(steps):
                if s == step:
                    out[:, k] = nxt
        return clip(out)


REFERENCE_MODELS = {"persistence": Persistence, "linear_extrapolation": LinearExtrapolation, "ar3": AutoRegressive}
