"""Stage 2 for one patient, anchor and model (script 30; HANDOFF 7.4, D80, D81): refit policies.

From the anchor (day 14 after day 0, D80) every policy predicts the same
``test_days`` days. A policy says when the model is refitted and on which
days:

- **window** (``Window``): ``growing`` (every day from the anchor - 14 days to
  the refit, as in stage 1), ``rollingN`` (the last N days before the refit,
  never earlier than the anchor - 14 days) or ``expN`` (the growing window
  with sample weights ``0.5 ** (age in days / N)``: recent days weigh more);
- **frequency**: every 1, 3, 7 or 14 days, or ``never`` (the fit at the
  anchor only, on the 14 days before it);
- **triggers** (D81): refit the day after a drift detector (ADWIN or
  Page-Hinkley, river) flags that the model has become worse *relative to
  persistence*; and the matched fixed policy (``even_<trigger>``) with the same
  number of refits, evenly spread over the test days;
- **online** (ARX only, D82): the update at every sample of its mechanism;
  each forecast uses every sample (growing window, forgetting factor) whose
  targets are known by its origin.

Every refit starts again from the model's starting point (D65): the learned
models are fine-tuned from their population model, AR(3) and ARX are
re-estimated. A refit at the start of day *t* learns only from samples whose
targets are before *t* (HANDOFF 7.1).

Fits are shared between the policies of one job (``FitStore``): a fit is
defined by its training days, its day and its weights, so ``growing`` every 7
days reuses the fits of ``growing`` every day, and ``growing``, ``rolling14``,
``rolling28`` and ``never`` share the fit at the anchor (``rolling28`` shares
every fit of the first 14 days, when its window still reaches the start). Each fit forecasts every test origin
from its day on once; each policy takes the part it needs. The cost of a
policy is still counted as if it ran alone (every fit it uses).

The drift stream is causal: after day *d* the detector receives, for every
origin whose targets (up to the longest horizon) are all before day *d* + 1,
the mean over horizons of (|model error| - |persistence error|) / 50 mg/dL,
clipped to [-1, 1]. After a refit the detector starts again.

A refit that fails (for example a rolling window with no usable samples) keeps
the model in use, as a deployed system would: its forecasts come from the last
fit that worked (``served_by`` in the fit log), and at the anchor from the fit
of ``never``. If the fit of ``never`` fails, no model exists at the anchor: the
patient and anchor are left out for that model (``excluded`` in the fit log),
so that every policy of a model is scored on the same patients and origins.
"""

from __future__ import annotations

import math
import time
import traceback
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..data.store import ProcessedStore
from ..eval.metrics import ClarkeAccumulator
from ..models.base import GLUCOSE_SCALE, TrainData
from .ageing import SeriesWindow
from .samples import Samples, known_before, make_samples
from .stage0 import (CONTEXT_MODELS, ORIGIN_RULES, add_loop_columns, context_for, day_metrics, input_columns,
                     is_learned, make_model, split_name, validation_mask)

CACHE_VERSION = 2  # bump when the simulation changes, so that script 30 recomputes its cached results
TRIGGERS = ("adwin", "page_hinkley")
ONLINE = "online"        # the policy of the models that update at every sample
ONLINE_MODELS = ("arx",)  # models with ``online_predict``


@dataclass(frozen=True)
class Window:
    kind: str      # growing | rolling | exponential
    days: int = 0  # rolling length or half-life in days

    @property
    def key(self) -> str:
        return {"growing": "growing", "rolling": f"rolling{self.days}", "exponential": f"exp{self.days}"}[self.kind]

    def start(self, fit_day: pd.Timestamp, base_start: pd.Timestamp) -> pd.Timestamp:
        if self.kind == "rolling":
            return max(base_start, fit_day - pd.Timedelta(days=self.days))
        return base_start

    def weights(self, days: np.ndarray, fit_day: pd.Timestamp) -> np.ndarray | None:
        """Sample weights of the exponential window (None for the others); ``days`` as datetime64."""
        if self.kind != "exponential":
            return None
        age = (np.datetime64(fit_day.date()) - np.asarray(days).astype("datetime64[D]")).astype(float)
        return np.power(0.5, np.clip(age, 0, None) / self.days)


def parse_window(text: str) -> Window:
    if text == "growing":
        return Window("growing")
    for prefix, kind in (("rolling", "rolling"), ("exp", "exponential")):
        if text.startswith(prefix) and text[len(prefix):].isdigit():
            return Window(kind, int(text[len(prefix):]))
    raise ValueError(f"unknown window {text!r}: growing, rolling<days> or exp<half-life days>")


@dataclass(frozen=True)
class Policy:
    name: str
    window: Window
    every: int | None = None          # days between refits; None: never (or set by a trigger)
    trigger: str | None = None        # adwin | page_hinkley
    matched: str | None = None        # an "even" policy: the trigger whose number of refits it copies
    online: bool = False              # update at every sample (ARX)

    @property
    def kind(self) -> str:
        if self.online:
            return "online"
        return "trigger" if self.trigger else "matched" if self.matched else "never" if self.every is None else "fixed"


BASE = Window("growing")


def policy_name(window: Window, every: int | None) -> str:
    return "never" if every is None else f"{window.key}_every{every}"


def policy_grid(windows: list[str], every: list[int]) -> list[Policy]:
    """``never`` plus every window x frequency (D80)."""
    grid = [Policy("never", Window("growing"))]
    for w in windows:
        window = parse_window(w)
        grid += [Policy(policy_name(window, int(k)), window, int(k)) for k in every]
    return grid


def parse_policy(name: str) -> Policy:
    """The policy of a name made by ``policy_grid`` or a trigger name (``adwin_growing``, ``even_adwin_growing``)."""
    if name == "never":
        return Policy("never", Window("growing"))
    if name == ONLINE:
        return Policy(ONLINE, Window("growing"), online=True)
    if name.startswith("even_"):
        trigger = parse_policy(name[len("even_"):])
        return Policy(name, trigger.window, matched=trigger.trigger)
    for t in TRIGGERS:
        if name.startswith(t + "_"):
            return Policy(name, parse_window(name[len(t) + 1:]), trigger=t)
    window, _, every = name.rpartition("_every")
    return Policy(name, parse_window(window), int(every))


def online_policies(model: str) -> list[str]:
    """``online`` for the models that update at every sample (D82), nothing for the others."""
    return [ONLINE] if split_name(model)[0] in ONLINE_MODELS else []


def trigger_policies(triggers: list[str], window: str) -> list[Policy]:
    w = parse_window(window)
    out = []
    for t in triggers:
        out += [Policy(f"{t}_{w.key}", w, trigger=t), Policy(f"even_{t}_{w.key}", w, matched=t)]
    return out


def fixed_fit_days(policy: Policy, anchor: pd.Timestamp, test_days: int) -> list[pd.Timestamp]:
    if policy.every is None:
        return [anchor]
    return [anchor + pd.Timedelta(days=k) for k in range(0, int(test_days), int(policy.every))]


def even_fit_days(refits: int, anchor: pd.Timestamp, test_days: int) -> list[pd.Timestamp]:
    """The fit at the anchor and ``refits`` more, evenly spread over the test days."""
    offsets = sorted({int(round(k * test_days / (refits + 1))) for k in range(1, refits + 1)} - {0})
    return [anchor] + [anchor + pd.Timedelta(days=o) for o in offsets if o < test_days]


def make_detector(kind: str, options: dict):
    from river import drift

    if kind == "adwin":
        return drift.ADWIN(**options)
    if kind == "page_hinkley":
        return drift.PageHinkley(**options)
    raise ValueError(f"unknown trigger {kind!r}; one of {TRIGGERS}")


@dataclass
class PolicyResult:
    patient_id: str
    period: str
    anchor_day: int
    model: str
    days: pd.DataFrame
    clarke: ClarkeAccumulator | None
    info: list = field(default_factory=list)
    refits: dict = field(default_factory=dict)
    seconds: float = 0.0
    error: str = ""
    warnings: str = ""


class FitStore:
    """The fits of one job, one per (window, fit day), each with its forecasts of every test origin from its day."""

    def __init__(self, name, seed, settings, device, test, pool, series, base_start) -> None:
        self.name, self.seed, self.settings, self.device = name, seed, settings, device
        self.test, self.pool, self.series, self.base_start = test, pool, series, base_start
        self.test_day = pd.DatetimeIndex(test.day)
        self.fits: dict = {}

    def get(self, window: Window, fit_day: pd.Timestamp) -> dict:
        start = window.start(fit_day, self.base_start)
        key = (start, fit_day, window.key if window.kind == "exponential" else "")  # same data, same fit
        if key in self.fits:
            return self.fits[key]
        info = {"fit_day": fit_day.date(), "train_start": start.date(),
                "weighting": window.key if window.kind == "exponential" else "none"}
        rows = np.flatnonzero(self.test_day >= fit_day)
        entry = {"rows": rows, "pred": None, "info": info}
        try:
            in_window = (self.pool.day >= np.datetime64(start.date())) & (self.pool.day < np.datetime64(fit_day.date()))
            train = self.pool.subset(in_window)
            train = train.subset(known_before(train, fit_day))
            values, segment, step = self.series.between(start, fit_day)
            data = TrainData(train, values, segment, step,
                             validation=validation_mask(train.day, int(self.settings["validation_days"])),
                             sample_weight=window.weights(train.day, fit_day),
                             series_weight=window.weights(self.series.day, fit_day))
            info.update(train_days=int(len(np.unique(train.day))), train_samples=int(train.n))
            model = make_model(self.name, int(self.seed), self.settings, self.device)
            log = model.timed_fit(data)
            begin = time.time()
            entry["pred"] = model.predict(self.test.subset(rows)) if len(rows) else np.zeros((0, 0), np.float32)
            details = {k: (str(v) if isinstance(v, (list, dict)) else v) for k, v in log.details.items()
                       if k not in ("curve", "compute_seconds")}
            info.update(fit_samples=log.n_samples, fit_seconds=log.seconds,
                        compute_seconds=float(log.details.get("compute_seconds", log.seconds)),
                        predict_seconds=round(time.time() - begin, 3), **details,
                        predict_device=getattr(model, "device", None))
            del model
        except Exception as error:  # noqa: BLE001
            info["error"] = f"{type(error).__name__}: {str(error).strip()}"[:300]
        self.fits[key] = entry
        return entry

    def forecast(self, window: Window, fit_day: pd.Timestamp, rows: np.ndarray) -> np.ndarray | None:
        """Forecasts of the fit (window, fit_day) for test rows on or after its day (None if the fit failed)."""
        entry = self.get(window, fit_day)
        if entry["pred"] is None:
            return None
        where = np.searchsorted(entry["rows"], rows)
        return entry["pred"][where]

    def serving(self, window: Window, fit_days: list[pd.Timestamp]) -> tuple[Window, pd.Timestamp]:
        """The fit in use after the refits ``fit_days``: the last one that worked; the fit of never at the first
        day (the anchor) if none did."""
        for day in reversed(fit_days):
            if self.get(window, day)["pred"] is not None:
                return window, day
        return BASE, fit_days[0]


def drift_values(test, pred: np.ndarray, reference: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Per origin: mean over horizons of (|model error| - |persistence error|) / 50, clipped to [-1, 1]."""
    targets = test.targets[rows]
    diff = (np.abs(pred - targets) - np.abs(reference[rows] - targets)) / GLUCOSE_SCALE
    ok = np.isfinite(diff)
    count = ok.sum(axis=1)
    value = np.where(ok, diff, 0.0).sum(axis=1) / np.maximum(count, 1)
    return np.where(count > 0, np.clip(value, -1.0, 1.0), np.nan)


def run_trigger(policy: Policy, store: FitStore, reference: np.ndarray, anchor: pd.Timestamp, test_days: int,
                options: dict) -> list[pd.Timestamp]:
    """The fit days chosen by a drift detector (the fit at the anchor first)."""
    test = store.test
    origin = pd.DatetimeIndex(test.origin_ts)
    complete = origin + pd.Timedelta(minutes=int(max(test.horizons_min)))  # all targets known from this time
    fits = [anchor]
    detector = make_detector(policy.trigger, options)
    fed = np.zeros(test.n, dtype=bool)
    for k in range(int(test_days) - 1):
        day_end = anchor + pd.Timedelta(days=k + 1)
        rows = np.flatnonzero((store.test_day >= fits[-1]) & (complete < day_end) & ~fed)
        if not len(rows):
            continue
        fed[rows] = True
        pred = store.forecast(*store.serving(policy.window, fits), rows)
        for value in drift_values(test, pred, reference, rows):
            if np.isfinite(value):
                detector.update(float(value))
                if detector.drift_detected:
                    break
        if detector.drift_detected:
            fits.append(day_end)  # refit at the start of the next day; the detector starts again
            detector = make_detector(policy.trigger, options)
    return fits


class JobData:
    """The samples of one patient, period, anchor and model (stages 2 and 3 and the confirmation): test origins from
    the anchor, the training pool and series from ``window_days`` before it, persistence on the test origins, and
    the day metrics and Clarke data recorded so far. Origins and training samples are slots of the cohort's usable
    days (``job["usable"]``, a rule of ``ORIGIN_RULES``; the core cohort's ``open_insulin`` by default)."""

    def __init__(self, job: dict) -> None:
        self.patient_id, self.period = job["patient_id"], job["period"]
        self.anchor_day, self.name = int(job["anchor_day"]), job["model"]
        s = self.settings = job["settings"]
        grid = ProcessedStore(job["dataset"], root=job["derived_root"], stage="features").read_frame(
            "grid", self.patient_id)
        if grid.empty:
            raise LookupError("no feature grid")
        add_loop_columns(grid)
        if self.period == "whole":
            period_days = set(pd.DatetimeIndex(grid["day"].unique()))
        else:
            period_days = set(pd.DatetimeIndex(grid.loc[grid["period"] == self.period, "day"].unique()))
        self.anchor = pd.Timestamp(job["anchor"]).normalize()
        self.test_days = int(s["test_days"])
        self.base_start = self.anchor - pd.Timedelta(days=int(s["window_days"]))
        self.end = self.anchor + pd.Timedelta(days=self.test_days)
        span = {d for d in period_days if self.base_start <= d < self.end}
        common = dict(horizons_min=s["horizons_min"], lookback_h=s["lookback_hours"],
                      min_window_share=s["min_window_share"], recent_complete_min=s["recent_complete_min"],
                      origin_conditions=ORIGIN_RULES[job.get("usable", "open_insulin")],
                      exog_columns=input_columns(self.name, s))
        long_history = context_for([self.name], s) if self.name in CONTEXT_MODELS else {}  # M2 (script 43, D99)
        self.test = make_samples(grid, stride_min=s["origin_stride_min"],
                                 allowed_days={d for d in span if d >= self.anchor}, **common, **long_history)
        self.pool = make_samples(grid, stride_min=s["train_stride_min"], allowed_days=span,
                                 trajectory=is_learned(self.name), **common)
        self.grid, self.span, self.common = grid, span, common  # for other origins (stage 3b: before the anchor)
        self.series = SeriesWindow(grid, span)
        self.clarke = ClarkeAccumulator(["model", "policy", "horizon_min", "period"],
                                        sample_size=int(s["clarke_point_sample"]), stream=int(job["stream"]))
        from ..models.reference import Persistence

        self.reference = Persistence().predict(self.test)
        self.test_day = pd.DatetimeIndex(self.test.day)
        self.frames: list = []

    def train_data(self, start: pd.Timestamp, fit_day: pd.Timestamp, window: Window | None = None) -> TrainData:
        """The training data of a fit on ``fit_day`` from ``start`` (targets before the fit day, HANDOFF 7.1)."""
        in_window = (self.pool.day >= np.datetime64(start.date())) & (self.pool.day < np.datetime64(fit_day.date()))
        train = self.pool.subset(in_window)
        train = train.subset(known_before(train, fit_day))
        values, segment, step = self.series.between(start, fit_day)
        return TrainData(train, values, segment, step,
                         validation=validation_mask(train.day, int(self.settings["validation_days"])),
                         sample_weight=window.weights(train.day, fit_day) if window else None,
                         series_weight=window.weights(self.series.day, fit_day) if window else None)

    def record(self, policy: str, seed: int, fit_day: pd.Timestamp, rows: np.ndarray, pred: np.ndarray,
               model: str | None = None) -> None:
        part = self.test.subset(rows)
        table = day_metrics(pred, part.targets, part.day, part.horizons_min,
                            tuple(self.settings.get("event_thresholds", ())))
        if table.empty:
            return
        name = model or self.name
        for column, value in (("fit_day", fit_day), ("seed", seed), ("policy", policy), ("model", name)):
            table.insert(0, column, value)
        self.frames.append(table)
        for k, h in enumerate(part.horizons_min):
            self.clarke.add((name, policy, h, self.period), part.targets[:, k], pred[:, k])

    def record_reference(self) -> None:
        self.record("reference", 0, self.anchor, np.arange(self.test.n), self.reference, model="persistence")

    def days(self) -> pd.DataFrame:
        days = pd.concat(self.frames, ignore_index=True) if self.frames else pd.DataFrame()
        if not days.empty:
            days.insert(0, "anchor_day", self.anchor_day)
            days.insert(0, "period", self.period)
            days.insert(0, "patient_id", self.patient_id)
            day = pd.to_datetime(days["day"])
            days["age_days"] = (day - pd.to_datetime(days["fit_day"])).dt.days
            days["days_since_anchor"] = (day - self.anchor).dt.days
        return days

    def origins_before_anchor(self) -> Samples:
        """Forecast origins of the training window (the days before the anchor), on the test stride."""
        return make_samples(self.grid, stride_min=self.settings["origin_stride_min"],
                            allowed_days={d for d in self.span if d < self.anchor}, **self.common)

    def row(self, policy: str, seed: int, **extra) -> dict:
        return {"patient_id": self.patient_id, "period": self.period, "anchor_day": self.anchor_day,
                "model": self.name, "policy": policy, "seed": int(seed), **extra}


def evaluate_policies(job: dict) -> PolicyResult:
    """Every policy of one model for one patient and anchor (see the module docstring)."""
    started = time.time()
    patient_id, period, anchor_day, name = job["patient_id"], job["period"], int(job["anchor_day"]), job["model"]
    try:
        try:
            data = JobData(job)
        except LookupError as error:
            return PolicyResult(patient_id, period, anchor_day, name, pd.DataFrame(), None, error=str(error))
        s, test, pool, anchor = data.settings, data.test, data.pool, data.anchor
        base_start, end, test_days, test_day, reference = (data.base_start, data.end, data.test_days, data.test_day,
                                                           data.reference)
        info, refits = [], {}
        if job.get("with_reference", True):
            data.record_reference()

        policies = [parse_policy(p) for p in job["policies"]]
        ordered = [p for p in policies if p.kind in ("never", "fixed", "trigger")] + \
            [p for p in policies if p.kind == "matched"]  # an "even" policy needs its trigger's refits
        online = [p for p in policies if p.kind == "online"]
        for seed in (s["seeds"] if is_learned(name) else [0]):
            store = FitStore(name, seed, s, job.get("device", "cpu"), test, pool, data.series, base_start)
            base = store.get(BASE, anchor)
            if base["pred"] is None:  # no model at the anchor: this patient and anchor are left out for the model
                for policy in ordered:
                    info.append(data.row(policy.name, seed, test_origins=0, **base["info"],
                                         excluded="the fit of never at the anchor failed"))
                    refits[(policy.name, int(seed))] = []
                del store
                continue
            for policy in online:  # the update at every sample: always the newest coefficients (age 0)
                refits[(policy.name, int(seed))] = [anchor]
                info.append(run_online(data, policy.name, seed, make_model(name, int(seed), s, "cpu")))
            for policy in ordered:
                if policy.kind == "trigger":
                    fit_days = run_trigger(policy, store, reference, anchor, test_days,
                                           dict(s["triggers"][policy.trigger]))
                elif policy.kind == "matched":
                    source = next((p.name for p in policies if p.trigger == policy.matched and
                                   p.window == policy.window), None)
                    count = len(refits.get((source, int(seed)), [anchor])) - 1
                    fit_days = even_fit_days(count, anchor, test_days)
                else:
                    fit_days = fixed_fit_days(policy, anchor, test_days)
                refits[(policy.name, int(seed))] = fit_days
                for i, fit_day in enumerate(fit_days):
                    stop = fit_days[i + 1] if i + 1 < len(fit_days) else end
                    rows = np.flatnonzero((test_day >= fit_day) & (test_day < stop))
                    if not len(rows):
                        continue
                    entry = store.get(policy.window, fit_day)
                    window, used = store.serving(policy.window, fit_days[: i + 1])
                    kept = {"served_by": f"{window.key} {used.date()}"} if (window, used) != (policy.window, fit_day) \
                        else {}
                    info.append(data.row(policy.name, seed, test_origins=int(len(rows)), **entry["info"], **kept))
                    data.record(policy.name, int(seed), used, rows, store.forecast(window, used, rows))
            del store
        refits = {f"{p}|{seed}": [d.date().isoformat() for d in v] for (p, seed), v in refits.items()}
        return PolicyResult(patient_id, period, anchor_day, name, data.days(), data.clarke, info, refits,
                            round(time.time() - started, 2))
    except Exception:  # noqa: BLE001
        return PolicyResult(patient_id, period, anchor_day, name, pd.DataFrame(), None,
                            seconds=round(time.time() - started, 2), error=traceback.format_exc(limit=6))


def run_online(data: JobData, policy: str, seed: int, model) -> dict:
    """The update at every sample over the growing window from ``base_start`` (D82): forecasts recorded per day
    (always the newest coefficients, age 0); returns the fit-log row."""
    begin = time.time()
    row = data.row(policy, seed, test_origins=int(data.test.n), fit_day=data.anchor.date(),
                   train_start=data.base_start.date(), weighting="online")
    try:
        in_window = data.pool.subset(data.pool.day >= np.datetime64(data.base_start.date()))
        pred, updates = model.online_predict(in_window, data.test)
        seconds = round(time.time() - begin, 3)
        row.update(train_samples=int(in_window.n), updates=updates, fit_seconds=seconds, compute_seconds=seconds)
        for day in np.unique(data.test.day):
            rows = np.flatnonzero(data.test.day == day)
            data.record(policy, int(seed), pd.Timestamp(day), rows, pred[rows])
    except Exception as error:  # noqa: BLE001
        row["error"] = f"{type(error).__name__}: {str(error).strip()}"[:300]
    return row


def refits_per_week(fit_days: list, test_days: int) -> float:
    """Refits after the first fit, per week of test days."""
    return 7.0 * max(0, len(fit_days) - 1) / max(1, test_days)


def halving_keep(n_policies: int, share: float = 1 / 3) -> int:
    return max(1, int(math.ceil(n_policies * share)))
