"""Stage 1 for one patient and anchor (script 20; HANDOFF 7.3, D65): how a personal model ages.

From an anchor day *a* (14, 42, 70 or 98 days after day 0, D51) every model is
tested on the same days, the ``test_days`` days from *a*, in three versions:

- ``frozen``: fitted once at *a* on the 14 days before it;
- ``rolling``: refitted every ``refresh`` days on the 14 days before the refit;
- ``growing``: refitted every ``refresh`` days on every day from *a* - 14 up to
  the refit (same start as the frozen window, so frozen and growing share the
  first fit).

Frozen against rolling measures the effect of recency (same amount of data,
older or newer); rolling against growing measures the effect of the amount of
data (both fresh). Every refit starts again from the model's starting point
(D63): a learned model fine-tuned from the population model, or AR(3)
re-estimated. A refit at the start of day *t* learns only from samples whose
targets are before *t* (HANDOFF 7.1), with the last 2 days of its window that
have samples for early stopping (D62).

Returned per (model, version, seed, horizon, test day): number of forecasts,
sums of errors and Clarke zone counts, the day of the fit that made the
forecast and its age (days since that fit), plus persistence on the same
origins (the reference of the daily skill).
"""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..data.store import ProcessedStore
from ..eval.metrics import ClarkeAccumulator
from ..models.base import EXOG, TrainData
from .samples import DenseGrid, known_before, make_samples
from .stage0 import (ORIGIN_CONDITIONS, add_loop_columns, day_metrics, is_learned, make_model, needs_exog,
                     per_model, validation_mask)

VERSIONS = ("frozen", "rolling", "growing")
CACHE_VERSION = 1  # bump when the simulation changes, so that script 20 recomputes its cached results


@dataclass(frozen=True)
class Fit:
    version: str
    fit_day: pd.Timestamp     # the refit happens at the start of this day, which is the first day it predicts
    train_start: pd.Timestamp  # first training day (inclusive)
    test_end: pd.Timestamp    # first day it no longer predicts (exclusive)


def plan_fits(anchor: pd.Timestamp, test_days: int, window_days: int, every: int, version: str) -> list[Fit]:
    """The fits of one version: when each is made, which days it learns from and which days it predicts."""
    anchor = pd.Timestamp(anchor).normalize()
    end = anchor + pd.Timedelta(days=int(test_days))
    window = pd.Timedelta(days=int(window_days))
    if version == "frozen":
        return [Fit(version, anchor, anchor - window, end)]
    if version not in VERSIONS:
        raise ValueError(f"version must be one of {VERSIONS}")
    fits, t, step = [], anchor, pd.Timedelta(days=max(1, int(every)))
    while t < end:
        fits.append(Fit(version, t, t - window if version == "rolling" else anchor - window, min(t + step, end)))
        t += step
    return fits


@dataclass
class AgeingResult:
    patient_id: str
    period: str
    anchor_day: int
    days: pd.DataFrame
    clarke: ClarkeAccumulator | None
    info: list = field(default_factory=list)
    seconds: float = 0.0
    error: str = ""
    warnings: str = ""


class SeriesWindow:
    """The patient's glucose input on contiguous slots, masked to any range of days (for AR models)."""

    def __init__(self, grid: pd.DataFrame, allowed: set) -> None:
        dense = DenseGrid(grid)
        self.values = dense.column("glucose_input")
        self.segment = dense.segment
        self.step = dense.step_min
        self.day = dense.slot_times().astype("datetime64[D]")
        allowed_days = np.array(sorted(pd.DatetimeIndex(list(allowed)).values.astype("datetime64[D]")))
        self.allowed = np.isin(self.day, allowed_days)

    def between(self, start: pd.Timestamp, end: pd.Timestamp) -> tuple[np.ndarray, np.ndarray, float]:
        keep = self.allowed & (self.day >= np.datetime64(start.date())) & (self.day < np.datetime64(end.date()))
        return np.where(keep, self.values, np.nan), self.segment, self.step


def evaluate_anchor(job: dict) -> AgeingResult:
    """Every model and version of one patient from one anchor (see the module docstring)."""
    started = time.time()
    patient_id, period, anchor_day = job["patient_id"], job["period"], int(job["anchor_day"])
    try:
        s = job["settings"]
        grid = ProcessedStore(job["dataset"], root=job["derived_root"], stage="features").read_frame("grid", patient_id)
        if grid.empty:
            return AgeingResult(patient_id, period, anchor_day, pd.DataFrame(), None, error="no feature grid")
        add_loop_columns(grid)
        if period == "whole":
            period_days = set(pd.DatetimeIndex(grid["day"].unique()))
        else:
            period_days = set(pd.DatetimeIndex(grid.loc[grid["period"] == period, "day"].unique()))
        anchor = pd.Timestamp(job["anchor"]).normalize()
        window = pd.Timedelta(days=int(s["window_days"]))
        end = anchor + pd.Timedelta(days=int(s["test_days"]))
        span = {d for d in period_days if anchor - window <= d < end}
        test_days = {d for d in span if d >= anchor}
        learned = any(is_learned(m) for m in s["models"])
        common = dict(horizons_min=s["horizons_min"], lookback_h=s["lookback_hours"],
                      min_window_share=s["min_window_share"], recent_complete_min=s["recent_complete_min"],
                      origin_conditions=ORIGIN_CONDITIONS,
                      exog_columns=EXOG if any(needs_exog(m) for m in s["models"]) else ())
        test = make_samples(grid, stride_min=s["origin_stride_min"], allowed_days=test_days, **common)
        # every training sample any fit may use: the window before the anchor and the test days (later refits learn
        # from the test days already seen)
        pool = make_samples(grid, stride_min=s["train_stride_min"], allowed_days=span, trajectory=learned, **common)
        series = SeriesWindow(grid, span)
        clarke = ClarkeAccumulator(["model", "version", "horizon_min", "period"],
                                   sample_size=int(s["clarke_point_sample"]), stream=int(job["stream"]))
        test_day = pd.DatetimeIndex(test.day)
        frames, info = [], []

        def record(name: str, version: str, seed: int, fit_day: pd.Timestamp, mask: np.ndarray, pred: np.ndarray):
            part = test.subset(mask)
            table = day_metrics(pred, part.targets, part.day, part.horizons_min)
            if table.empty:
                return
            table.insert(0, "fit_day", fit_day)
            table.insert(0, "seed", seed)
            table.insert(0, "version", version)
            table.insert(0, "model", name)
            frames.append(table)
            for k, h in enumerate(part.horizons_min):
                clarke.add((name, version, h, period), part.targets[:, k], pred[:, k])

        from ..models.reference import Persistence

        everything = np.ones(test.n, dtype=bool)
        record("persistence", "reference", 0, anchor, everything, Persistence().predict(test))
        for name in s["models"]:
            if name == "persistence":
                continue
            every = int(per_model(s["refresh_days"], name, 1))  # a glucose-only twin as its full model (D90)
            for seed in (s["seeds"] if is_learned(name) else [0]):
                for version in s["versions"]:
                    for fit in plan_fits(anchor, int(s["test_days"]), int(s["window_days"]), every, version):
                        mask = (test_day >= fit.fit_day) & (test_day < fit.test_end)
                        if not mask.any():
                            continue  # nothing to predict: no refit needed
                        base = {"patient_id": patient_id, "period": period, "anchor_day": anchor_day, "model": name,
                                "version": version, "seed": int(seed), "fit_day": fit.fit_day.date(),
                                "train_start": fit.train_start.date(), "test_origins": int(mask.sum())}
                        in_window = (pool.day >= np.datetime64(fit.train_start.date())) & \
                            (pool.day < np.datetime64(fit.fit_day.date()))
                        train = pool.subset(in_window)
                        train = train.subset(known_before(train, fit.fit_day))
                        data = TrainData(train, *series.between(fit.train_start, fit.fit_day),
                                         validation=validation_mask(train.day, int(s["validation_days"])))
                        try:
                            model = make_model(name, int(seed), s, job.get("device", "cpu"))
                            log = model.timed_fit(data)
                            begin = time.time()
                            pred = model.predict(test.subset(mask))
                        except Exception as error:  # noqa: BLE001
                            info.append({**base, "train_samples": train.n,
                                         "error": f"{type(error).__name__}: {str(error).strip()}"[:300]})
                            continue
                        record(name, version, int(seed), fit.fit_day, mask, pred)
                        details = {k: (str(v) if isinstance(v, (list, dict)) else v) for k, v in log.details.items()
                                   if k != "curve"}
                        info.append({**base, "train_days": int(len(np.unique(train.day))), "train_samples": train.n,
                                     "fit_samples": log.n_samples, "fit_seconds": log.seconds,
                                     "predict_seconds": round(time.time() - begin, 3), **details,
                                     "predict_device": getattr(model, "device", None)})
        days = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if not days.empty:
            days.insert(0, "anchor_day", anchor_day)
            days.insert(0, "period", period)
            days.insert(0, "patient_id", patient_id)
            day = pd.to_datetime(days["day"])
            days["age_days"] = (day - pd.to_datetime(days["fit_day"])).dt.days
            days["days_since_anchor"] = (day - anchor).dt.days
        return AgeingResult(patient_id, period, anchor_day, days, clarke, info, round(time.time() - started, 2))
    except Exception:  # noqa: BLE001
        return AgeingResult(patient_id, period, anchor_day, pd.DataFrame(), None, seconds=round(time.time() - started, 2),
                            error=traceback.format_exc(limit=6))
