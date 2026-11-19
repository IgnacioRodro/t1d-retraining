"""ARX with a forgetting factor (HANDOFF 6.1, M02; D69): the "update at every sample" mechanism.

One linear model per horizon predicts the change from the glucose at the
origin from:

- the glucose at the origin, (glucose - 140) / 50 (so that high values can be
  pulled down and low values up);
- the changes of the last hour, (glucose 5, 10, 15, 30 and 60 min before -
  glucose at the origin) / 50, missing inputs carried forward;
- IOB, COB, insulin activity and carbohydrate absorption at the origin
  (scaled as for the networks; left out with ``exog=False``, the glucose-only
  twin of D76 and D90) and an intercept.

The coefficients are the recursive least squares solution with forgetting
factor ``forgetting`` over the training samples in time order, written in
closed form: each sample weighs ``forgetting ** (age in 5-min steps)``
relative to the newest one, plus a small ridge penalty. The effective memory is
about 1 / (1 - forgetting) steps (0.999: about 3.5 days). The default is no
forgetting (λ = 1, D87): stage 3 found that ARX gains with every day of memory,
so the window of each refit policy sets the memory, as for the other models;
stage 3 still tries smaller factors in the online update. Fitting is cheap, so
stage 1 refits it every day. The sample-by-sample online update (Eren-Oruklu
et al. 2009) is the same recursion run on the test days (``online_predict``,
the ``online`` policy of stage 2, D82): every forecast uses the solution over
every sample whose targets are known by its origin, the same as ``fit`` on
those samples.

Two options of the later analyses: ``low_loss`` (script 71, D96), weighted
least squares in which every target below ``threshold`` mg/dL weighs
``factor`` times (in ``fit`` and in the online update), and ``extra`` (script
73, D98), the wearables at the origin as more inputs, centred and scaled with
fixed constants (``WEARABLE_INPUTS``; a missing value counts as the centre).
"""

from __future__ import annotations

import numpy as np

from ..sim.samples import Samples
from .base import EXOG, EXOG_SCALE, GLUCOSE_SCALE, WEARABLE_INPUTS, FitLog, ForecastModel, TrainData, clip, low_weights

GLUCOSE_CENTER = 140.0
LAGS_MIN = (5, 10, 15, 30, 60)
INPUTS = ("iob", "cob", "insulin_activity", "carb_absorption")


def arx_features(s: Samples, exog: bool = True, extra: tuple[str, ...] = ()) -> np.ndarray:
    x = np.asarray(s.inputs, dtype=float)
    filled = x.copy()
    for j in range(1, filled.shape[1]):  # carry the last reading forward
        gap = ~np.isfinite(filled[:, j])
        filled[gap, j] = filled[gap, j - 1]
    g0 = x[:, -1]
    last = x.shape[1] - 1
    columns = [(g0 - GLUCOSE_CENTER) / GLUCOSE_SCALE]
    for minutes in LAGS_MIN:
        k = min(last, int(round(minutes / s.step_min)))
        columns.append(np.nan_to_num((filled[:, last - k] - g0) / GLUCOSE_SCALE, nan=0.0))
    for name in INPUTS if exog else ():
        columns.append(np.nan_to_num(np.asarray(s.exog[name][:, -1], dtype=float) / EXOG_SCALE[name], nan=0.0))
    for name in extra:
        center, scale = WEARABLE_INPUTS[name]
        columns.append(np.nan_to_num((np.asarray(s.exog[name][:, -1], dtype=float) - center) / scale, nan=0.0))
    columns.append(np.ones(len(g0)))
    return np.column_stack(columns)


class RecursiveARX(ForecastModel):
    name = "arx"
    mechanism = "sample_wise"
    exog = EXOG

    def __init__(self, forgetting: float = 1.0, ridge: float = 1.0, exog: bool = True, low_loss: dict | None = None,
                 extra: tuple[str, ...] = ()) -> None:
        if not 0 < forgetting <= 1:
            raise ValueError("forgetting must be in (0, 1]")
        unknown = [name for name in extra if name not in WEARABLE_INPUTS]
        if unknown:
            raise ValueError(f"no scale for the extra inputs {unknown} (models.base.WEARABLE_INPUTS)")
        self.forgetting = float(forgetting)
        self.ridge = float(ridge)
        self.use_exog = bool(exog)
        self.low_loss = dict(low_loss) if low_loss else None
        self.extra = tuple(extra)
        self.coef: np.ndarray | None = None

    def penalty(self, size: int) -> np.ndarray:
        penalty = self.ridge * np.eye(size)
        penalty[-1, -1] = 0.0  # the intercept is not penalised
        return penalty

    def fit(self, train: TrainData) -> FitLog:
        s = train.samples
        if not s.n:
            raise ValueError("no training samples")
        x = arx_features(s, self.use_exog, self.extra)
        g0 = np.asarray(s.inputs[:, -1], dtype=float)
        t = np.asarray(s.origin_ts).astype("datetime64[ns]").astype(np.int64) / 60e9
        age_steps = (t.max() - t) / s.step_min
        low = low_weights(s.targets, self.low_loss)
        weight = self.forgetting ** age_steps
        if train.sample_weight is not None:  # stage 2, exponential window: on top of the forgetting factor
            weight = weight * np.asarray(train.sample_weight, dtype=float)
        penalty = self.penalty(x.shape[1])
        coef = []
        for k in range(s.targets.shape[1]):
            y = (np.asarray(s.targets[:, k], dtype=float) - g0) / GLUCOSE_SCALE
            ok = np.isfinite(y)
            if ok.sum() < x.shape[1]:
                raise ValueError(f"too few training samples with a target at {s.horizons_min[k]} min")
            w = weight[ok] * (low[ok, k] if low is not None else 1.0)
            a = x[ok] * w[:, None]
            coef.append(np.linalg.solve(x[ok].T @ a + penalty, a.T @ y[ok]))
        self.coef = np.column_stack(coef)
        return FitLog(n_samples=int(s.n), details={"effective_memory_steps": round(1 / max(1e-9, 1 - self.forgetting))})

    def predict(self, samples: Samples) -> np.ndarray:
        g0 = np.asarray(samples.inputs[:, -1], dtype=float)
        return clip(g0[:, None] + GLUCOSE_SCALE * (arx_features(samples, self.use_exog, self.extra) @ self.coef))

    def online_predict(self, train: Samples, test: Samples, cold: bool = False) -> tuple[np.ndarray, int]:
        """Forecasts of every test origin with the coefficients of every training sample whose targets (up to the
        longest horizon) are earlier than the origin: the recursive least squares update with forgetting factor,
        sample by sample (stage 2, D82). Returns the forecasts and the number of distinct coefficient updates.
        ``cold``: origins with too few earlier samples get NaN instead of an error (stage 3b scores the days before
        the anchor too)."""
        order = np.argsort(np.asarray(train.origin_ts).astype("datetime64[ns]"), kind="stable")
        s = train.subset(order)
        if not s.n:
            raise ValueError("no training samples")
        x = arx_features(s, self.use_exog, self.extra)
        g0 = np.asarray(s.inputs[:, -1], dtype=float)
        t = np.asarray(s.origin_ts).astype("datetime64[ns]").astype(np.int64) / 60e9
        known = t + max(s.horizons_min)  # minute when every target of the sample is known
        t_test = np.asarray(test.origin_ts).astype("datetime64[ns]").astype(np.int64) / 60e9
        count = np.searchsorted(known, t_test, side="left")  # samples known strictly before each test origin
        needed = np.unique(count)
        penalty = self.penalty(x.shape[1])
        y = (np.asarray(s.targets, dtype=float) - g0[:, None]) / GLUCOSE_SCALE
        ok = np.isfinite(y)
        y = np.where(ok, y, 0.0)
        low = low_weights(s.targets, self.low_loss)
        w = ok * (low if low is not None else 1.0)  # weight of every sample and horizon (0: no target)
        horizons = y.shape[1]
        a = np.zeros((horizons, x.shape[1], x.shape[1]))
        b = np.zeros((horizons, x.shape[1]))
        seen = np.zeros(horizons, dtype=int)
        coef = {}
        outer = x[:, :, None] * x[:, None, :]
        step = max(1e-9, float(s.step_min))
        n = 0
        for target in needed:
            while n < target:  # add sample n; the older ones decay by the steps between them
                if n:
                    decay = self.forgetting ** ((t[n] - t[n - 1]) / step)
                    a *= decay
                    b *= decay
                a += w[n][:, None, None] * outer[n][None]
                b += (w[n] * y[n])[:, None] * x[n][None]
                seen += ok[n]
                n += 1
            if (seen < x.shape[1]).any():
                coef[int(target)] = None
                continue
            coef[int(target)] = np.column_stack([np.linalg.solve(a[k] + penalty, b[k]) for k in range(horizons)])
        if not cold and any(coef[int(c)] is None for c in count):
            raise ValueError("too few training samples before the first test origin")
        xt = arx_features(test, self.use_exog, self.extra)
        change = np.full((test.n, horizons), np.nan)
        for c in needed:
            rows = count == c
            if coef[int(c)] is not None:
                change[rows] = xt[rows] @ coef[int(c)]
        g_test = np.asarray(test.inputs[:, -1], dtype=float)
        return clip(g_test[:, None] + GLUCOSE_SCALE * change), int(len(needed))
