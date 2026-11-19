"""Stage 3b for one patient (script 42; HANDOFF 7.5, D89): insulin action duration and carbohydrate absorption.

The inputs IOB, insulin activity, COB and carbohydrate absorption (D54) come
from one curve for every patient: insulin acting over 360 min with its peak at
75 min, and carbohydrates absorbed linearly over 180 min. Stage 3b computes them
again from the delivered insulin and the logged carbohydrates with every
combination of ``PhysiologySet`` (action duration, peak, absorption time) and
asks whether a patient's own curve is better, and whether it can be found from
the patient's past data alone.

Arms, from the anchor of stage 3 over its test days (growing window):

- ARX, updated at every sample without forgetting (the best ARX of stage 3,
  D87):
  - ``fixed_<set>``: one set for everyone (the set of the settings is the
    default; the others show how sensitive the population choice is);
  - ``personal``: on every test day, the set whose forecasts were best so far
    for this patient (prequential skill, mean over horizons, on every earlier
    origin whose targets were already known, from the first day of the
    training window). It uses no future data. The best set on the test days
    themselves (an oracle, an upper bound only) is worked out by script 42 from
    the fixed sets.
- LightGBM fine-tuned, refitted from the population model at its optimal
  interval (restart, D65, the best mechanism of stage 3):
  - ``never`` and ``restart`` with the default set;
  - ``restart_personal``: every refit uses the set that ARX's prequential
    choice names on the refit day (a cheap linear model finds the curve, the
    nonlinear model uses it; the population model itself was trained with the
    default curve).

Every arm is scored on the same origins. The day metrics carry the model
(``arx`` or the LightGBM name) and the arm as ``policy``; the info rows are the
fits (``weighting`` = arm) and the daily choices (``chosen_set``).
"""

from __future__ import annotations

import copy
import dataclasses
import itertools
import time
import traceback
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..data.store import ProcessedStore
from ..features.build import evaluation_times, insulin_carb_inputs
from ..models.arx import RecursiveARX
from ..models.reference import Persistence
from .mechanisms import fit_days_of, log_row
from .policies import JobData, PolicyResult
from .samples import DenseGrid, Samples
from .stage0 import is_learned, make_model

CHANGED = ("iob", "insulin_activity", "cob", "carb_absorption")  # the model inputs that depend on the curves
LIGHTGBM_ARMS = ("never", "restart", "restart_personal")
CACHE_VERSION = 1  # bump when the simulation changes, so that script 42 recomputes its cached results


@dataclass(frozen=True)
class PhysiologySet:
    action_min: float
    peak_min: float
    absorption_min: float

    @property
    def name(self) -> str:
        return f"dia{self.action_min:g}_peak{self.peak_min:g}_carb{self.absorption_min:g}"

    def settings(self, base: dict) -> dict:
        return {**base, "insulin_action_min": self.action_min, "insulin_peak_min": self.peak_min,
                "carb_absorption_min": self.absorption_min}


def parameter_sets(actions, peaks, absorptions) -> list[PhysiologySet]:
    """Every combination; the exponential insulin curve needs its peak before half the action duration."""
    out = []
    for a, p, c in itertools.product(actions, peaks, absorptions):
        if not 0 < float(p) < float(a) / 2:
            raise ValueError(f"insulin peak {p} min must be below half the action duration {a} min")
        if float(c) <= 0:
            raise ValueError("the carbohydrate absorption time must be positive")
        out.append(PhysiologySet(float(a), float(p), float(c)))
    return out


def default_set(features: dict) -> PhysiologySet:
    return PhysiologySet(float(features["insulin_action_min"]), float(features["insulin_peak_min"]),
                         float(features["carb_absorption_min"]))


def physiology_columns(grid: pd.DataFrame, insulin: pd.DataFrame, carbs: pd.DataFrame, features: dict,
                       pset: PhysiologySet) -> dict[str, np.ndarray]:
    """The inputs of ``CHANGED`` on the grid's slots with the curves of ``pset`` (as script 05 makes them)."""
    columns, _ = insulin_carb_inputs(evaluation_times(grid), grid["ts"].min(), insulin, carbs, pset.settings(features))
    return {name: columns[name].astype(np.float32) for name in CHANGED}


def swap_exog(samples: Samples, grid: pd.DataFrame, columns: dict[str, np.ndarray]) -> Samples:
    """``samples`` with the input windows of ``columns`` (values on the rows of ``grid``) instead of their own."""
    dense = DenseGrid(grid)
    lookback = samples.inputs.shape[1]
    index = (np.asarray(samples.origin_slot) - dense.slot0)[:, None] + np.arange(-lookback + 1, 1)[None, :]
    exog = dict(samples.exog)
    for name, values in columns.items():
        full = np.full(dense.n, np.nan)
        full[dense.position] = values
        exog[name] = full[index].astype(np.float32) if samples.n else np.zeros((0, lookback), dtype=np.float32)
    return dataclasses.replace(samples, exog=exog)


def prequential_choice(forecasts: np.ndarray, persistence: np.ndarray, targets: np.ndarray, known: np.ndarray,
                       days: np.ndarray, default: int) -> tuple[np.ndarray, np.ndarray]:
    """For every day of ``days``, the set with the best skill (mean over horizons) on the origins whose targets
    were all known before the day began.

    ``forecasts`` (sets x origins x horizons), ``persistence`` and ``targets`` (origins x horizons), ``known`` the
    time at which the last target of each origin is known. An origin and horizon count only where every set has a
    forecast. Returns the chosen set per day (``default`` while nothing is known yet) and the score of every set
    per day (days x sets)."""
    ok = np.isfinite(targets) & np.isfinite(persistence) & np.isfinite(forecasts).all(axis=0)
    error = np.where(ok[None], (forecasts - targets[None]) ** 2, 0.0)          # sets x origins x horizons
    base = np.where(ok, (persistence - targets) ** 2, 0.0)                       # origins x horizons
    order = np.argsort(known, kind="stable")
    cum_error = np.concatenate([np.zeros((error.shape[0], 1, error.shape[2])),
                                np.cumsum(error[:, order], axis=1)], axis=1)
    cum_base = np.concatenate([np.zeros((1, base.shape[1])), np.cumsum(base[order], axis=0)], axis=0)
    count = np.searchsorted(known[order], days, side="left")  # origins known strictly before each day
    chosen = np.full(len(days), default, dtype=int)
    scores = np.full((len(days), error.shape[0]), np.nan)
    for i, c in enumerate(count):
        b = cum_base[c]
        usable = b > 0
        if not usable.any():
            continue
        skill = 1 - cum_error[:, c][:, usable] / b[usable][None]                # sets x horizons
        scores[i] = skill.mean(axis=1)
        best = np.flatnonzero(scores[i] >= scores[i].max() - 1e-12)
        chosen[i] = default if default in best else int(best[0])
    return chosen, scores


def evaluate_physiology(job: dict) -> PolicyResult:
    """Every arm of stage 3b for one patient and anchor (see the module docstring). The job holds, besides the
    usual keys of ``JobData``: ``sets`` (``PhysiologySet``), ``features`` (the settings of script 05),
    ``lightgbm`` (the LightGBM model name, or None), ``interval`` (its refit interval) and ``device``."""
    started = time.time()
    patient_id, period, anchor_day = job["patient_id"], job["period"], int(job["anchor_day"])
    learned = job.get("lightgbm")
    name = learned or "arx"
    try:
        try:
            data = JobData({**job, "model": name})
        except LookupError as error:
            return PolicyResult(patient_id, period, anchor_day, name, pd.DataFrame(), None, error=str(error))
        s, features, sets = data.settings, job["features"], list(job["sets"])
        default = default_set(features)
        if default not in sets:
            sets.append(default)
        k_default = sets.index(default)
        source = ProcessedStore(job["dataset"], root=job["derived_root"])
        insulin, carbs = source.read("insulin", patient_id), source.read("carbs", patient_id)
        if job.get("with_reference", True):
            data.record_reference()
        before = data.origins_before_anchor()
        info, refits = [], {}

        # the inputs of every set, and how far the default set is from the saved grid (script 05)
        columns = {pset: physiology_columns(data.grid, insulin, carbs, features, pset) for pset in sets}
        stored = {c: data.grid[c].to_numpy(float) for c in CHANGED}
        gap = max(float(np.nanmax(np.abs(columns[default][c] - stored[c]), initial=0.0)) for c in CHANGED)
        views: dict = {}

        def view_of(pset: PhysiologySet) -> JobData:
            """The patient's samples with the inputs of ``pset`` (made when first needed; the recorded days and
            Clarke data are shared with ``data``)."""
            if pset not in views:
                view = copy.copy(data)
                view.pool = swap_exog(data.pool, data.grid, columns[pset])
                view.test = swap_exog(data.test, data.grid, columns[pset])
                views[pset] = view
            return views[pset]

        # ARX at every sample, every set: forecasts on the test days and on the days before the anchor
        forgetting = float(job.get("forgetting", 1.0))
        test_pred, before_pred = [], []
        for pset in sets:
            view = copy.copy(data)  # ARX only: these samples are dropped after its forecasts
            view.pool = swap_exog(data.pool, data.grid, columns[pset])
            view.test = swap_exog(data.test, data.grid, columns[pset])
            early = swap_exog(before, data.grid, columns[pset])
            arm = f"fixed_{pset.name}"
            begin = time.time()
            row = data.row(arm, 0, model="arx", fit_day=data.anchor.date(), train_start=data.base_start.date(),
                           weighting=arm, test_origins=int(view.test.n), physiology_set=pset.name,
                           default_gap=round(gap, 6))
            model = RecursiveARX(forgetting=forgetting, ridge=float((s.get("arx") or {}).get("ridge", 1.0)))
            try:
                pred, updates = model.online_predict(view.pool, view.test)
                early_pred, _ = model.online_predict(view.pool, early, cold=True) if early.n else \
                    (np.zeros((0, len(s["horizons_min"]))), 0)
                seconds = round(time.time() - begin, 3)
                row.update(train_samples=int(view.pool.n), updates=updates, fit_seconds=seconds,
                           compute_seconds=seconds)
            except Exception as error:  # noqa: BLE001
                row["error"] = f"{type(error).__name__}: {str(error).strip()}"[:300]
                pred = np.full((view.test.n, len(s["horizons_min"])), np.nan)
                early_pred = np.full((early.n, len(s["horizons_min"])), np.nan)
            info.append(row)
            test_pred.append(pred)
            before_pred.append(early_pred)
            refits[f"{arm}|0"] = [data.anchor.date().isoformat()]

        if any("error" in r for r in info):  # an ARX set failed: no comparison of sets for this patient
            for r in info:
                r.setdefault("excluded", "an ARX set failed")
            if learned:
                info += [data.row(arm, 0, fit_day=data.anchor.date(), weighting=arm, test_origins=0,
                                  excluded="an ARX set failed") for arm in LIGHTGBM_ARMS]
            return PolicyResult(patient_id, period, anchor_day, name, data.days(), data.clarke, info, refits,
                                round(time.time() - started, 2))

        test, early = data.test, before
        targets = np.vstack([early.targets, test.targets]).astype(float)
        persistence = np.vstack([Persistence().predict(early), data.reference]).astype(float)
        forecasts = np.stack([np.vstack([b, t]) for b, t in zip(before_pred, test_pred)]).astype(float)
        origin = np.concatenate([np.asarray(early.origin_ts), np.asarray(test.origin_ts)]).astype("datetime64[ns]")
        known = origin + np.timedelta64(int(max(s["horizons_min"])), "m")
        test_days = np.array(sorted(np.unique(test.day))).astype("datetime64[ns]")
        chosen, scores = prequential_choice(forecasts, persistence, targets, known, test_days, k_default)
        day_of = dict(zip(pd.DatetimeIndex(test_days), chosen))

        for k, pset in enumerate(sets):
            for day in np.unique(test.day):
                rows = np.flatnonzero(test.day == day)
                data.record(f"fixed_{pset.name}", 0, pd.Timestamp(day), rows, test_pred[k][rows], model="arx")
        test_day = pd.DatetimeIndex(test.day)
        for day, k in day_of.items():
            rows = np.flatnonzero(test_day == day)
            data.record("personal", 0, day, rows, test_pred[k][rows], model="arx")
        info.append(data.row("personal", 0, model="arx", fit_day=data.anchor.date(), weighting="personal",
                             test_origins=int(test.n), compute_seconds=float(sum(r.get("compute_seconds", 0.0)
                                                                                 for r in info))))
        refits["personal|0"] = [data.anchor.date().isoformat()]
        for i, (day, k) in enumerate(day_of.items()):
            info.append({"patient_id": patient_id, "period": period, "anchor_day": anchor_day, "model": "arx",
                         "policy": "personal", "choice_day": day.date(), "chosen_set": sets[k].name,
                         "chosen_score": round(float(scores[i, k]), 5) if np.isfinite(scores[i, k]) else np.nan,
                         "default_score": round(float(scores[i, k_default]), 5)
                         if np.isfinite(scores[i, k_default]) else np.nan})

        if learned:
            info += lightgbm_arms(data, view_of, sets, default, day_of, job, refits)
        return PolicyResult(patient_id, period, anchor_day, name, data.days(), data.clarke, info, refits,
                            round(time.time() - started, 2))
    except Exception:  # noqa: BLE001
        return PolicyResult(patient_id, period, anchor_day, name, pd.DataFrame(), None,
                            seconds=round(time.time() - started, 2), error=traceback.format_exc(limit=6))


def lightgbm_arms(data: JobData, view_of, sets: list, default: PhysiologySet, day_of: dict, job: dict,
                  refits: dict) -> list[dict]:
    """never, restart (default set) and restart_personal (ARX's choice on each refit day) for the LightGBM model.
    A fit at the anchor with the default set is shared (as in stage 3); a refit that fails keeps the model in use,
    and a failed fit at the anchor leaves the LightGBM arms out."""
    name, s = job["lightgbm"], data.settings
    seed = int(s["seeds"][0]) if is_learned(name) else 0
    device = job.get("device", "cpu")
    fit_days = fit_days_of(data.anchor, data.test_days, int(job["interval"]))
    by_day = {pd.Timestamp(d): sets[k] for d, k in day_of.items()}

    def choice(day: pd.Timestamp) -> PhysiologySet:
        known = [d for d in by_day if d <= day]
        return by_day[max(known)] if known else default

    plans = {"never": [(data.anchor, default)], "restart": [(d, default) for d in fit_days],
             "restart_personal": [(d, choice(d)) for d in fit_days]}
    rows_info, anchor_model = [], None
    for arm, plan in plans.items():
        used_days, current, current_set, used = [], None, default, data.anchor
        for i, (day, pset) in enumerate(plan):
            stop = plan[i + 1][0] if i + 1 < len(plan) else data.end
            rows = np.flatnonzero((data.test_day >= day) & (data.test_day < stop))
            if i == 0 and pset == default and anchor_model is not None:  # the shared fit at the anchor
                model, row = anchor_model
                rows_info.append({**row, "policy": arm, "test_origins": int(len(rows))})
                current, current_set, used = model, pset, day
                used_days.append(day)
            else:
                begin = time.time()
                view = view_of(pset)
                train = view.train_data(data.base_start, day)
                try:
                    model = make_model(name, seed, s, device)
                    log = model.timed_fit(train)
                    weighting = "anchor" if arm == "never" else arm
                    row = {**log_row(view, weighting, seed, day, train, log, rows, begin), "policy": arm,
                           "physiology_set": pset.name}
                    rows_info.append(row)
                    current, current_set, used = model, pset, day
                    used_days.append(day)
                    if arm == "never":
                        anchor_model = (model, row)
                except Exception as error:  # noqa: BLE001
                    message = f"{type(error).__name__}: {str(error).strip()}"[:300]
                    if arm == "never":  # no model at the anchor: every LightGBM arm is left out
                        return [data.row(a, seed, fit_day=day.date(), train_start=data.base_start.date(),
                                         weighting="anchor", test_origins=0, error=message,
                                         excluded="the fit at the anchor failed") for a in LIGHTGBM_ARMS]
                    rows_info.append(data.row(arm, seed, fit_day=day.date(), train_start=data.base_start.date(),
                                              weighting=arm, test_origins=int(len(rows)), physiology_set=pset.name,
                                              error=message,
                                              served_by=str(used_days[-1].date()) if used_days else "anchor"))
                    if current is None:  # the arm's own first fit failed: the fit of never serves
                        current, current_set = anchor_model[0], default
                    used = used_days[-1] if used_days else data.anchor
            if len(rows) and current is not None:
                data.record(arm, seed, used, rows, current.predict(view_of(current_set).test.subset(rows)))
        refits[f"{arm}|{seed}"] = [d.date().isoformat() for d in used_days]
    return rows_info
