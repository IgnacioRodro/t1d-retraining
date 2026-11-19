"""Common evaluation designs against the simulator, for one patient, anchor and model (script 93, D112).

The test origins are those of the simulator (every origin of the test days from
the anchor, as in script 40). Each origin can be forecast by:

- the simulator's **frozen** model: fitted at the anchor on the ``window_days``
  before it and never again;
- the simulator's **refitted** model: fitted at the anchor and again every
  ``refit_every`` days on the growing window, every refit from the starting
  point (the restart mechanism of stage 2 and the confirmation);
- a model fitted by one of three common evaluation **designs**, each on the
  patient's whole record (the training window and the test days):

  - ``random_sample``: a random share of the test origins is held out, and the
    model learns from every other sample of the record, among them the
    neighbours of the held-out origins (a slot apart, sharing almost all their
    inputs and targets);
  - ``random_day``: a random share of the test days is held out, and the model
    learns from every other day, before and after them;
  - ``chronological``: the last share of the record's days is held out, and the
    model learns only from the samples whose targets precede them (no leakage;
    one fit just before the held-out days, scored once).

Each design is scored on its own held-out origins, and so are the frozen and
refitted models and persistence, so that every comparison is paired on the same
origins. The simulator's models learn only from samples whose targets precede
the day of their fit (HANDOFF 7.1); the designs do what their name says.

The held-out origins depend on the patient and the design only (not on the model
or the seed), so that every model is scored on the same origins. Only models
that learn from the samples can be fitted on a record with holes (LightGBM and
ARX): the series-based ones (AR(3), the networks) need contiguous days.
"""

from __future__ import annotations

import math
import time
import traceback
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..eval.metrics import ClarkeAccumulator
from ..eval.stage0_report import stable_hash
from ..models.base import TrainData
from .policies import BASE, FitStore, JobData
from .samples import known_before
from .stage0 import day_metrics, make_model, split_name, validation_mask

SPLITS = ("random_sample", "random_day", "chronological")
ARMS = ("design", "frozen", "refitted", "persistence")
SAMPLE_FAMILIES = ("lightgbm", "arx")  # models that learn from the samples, not from a contiguous series
CACHE_VERSION = 1  # bump when the simulation changes, so that script 93 recomputes its cached results


@dataclass
class SplitResult:
    patient_id: str
    period: str
    anchor_day: int
    model: str
    days: pd.DataFrame
    clarke: ClarkeAccumulator | None
    info: list = field(default_factory=list)
    seconds: float = 0.0
    error: str = ""


def check_model(name: str) -> None:
    if split_name(name)[0] not in SAMPLE_FAMILIES:
        raise ValueError(f"{name}: the designs need a model that learns from the samples ({SAMPLE_FAMILIES})")


def split_rng(seed: int, patient_id: str, split: str) -> np.random.Generator:
    """The random stream of one patient and design (the same for every model and seed)."""
    return np.random.default_rng([int(seed), stable_hash(f"{patient_id}|{split}")])


def held_out(data: JobData, split: str, share: float, rng: np.random.Generator) -> dict:
    """The held-out test origins of a design (``held``, over ``data.test``), the training samples it keeps
    (``train``, over ``data.pool``) and, for the chronological design, the first held-out day (``cut``)."""
    test, pool = data.test, data.pool
    if split == "random_sample":
        held = rng.random(test.n) < share
        if not held.any() and test.n:
            held[rng.integers(test.n)] = True
        return {"held": held, "train": ~np.isin(pool.origin_ts, test.origin_ts[held]), "cut": None}
    if split == "random_day":
        days = np.unique(test.day)
        chosen = rng.choice(days, size=max(1, int(round(share * len(days)))), replace=False)
        return {"held": np.isin(test.day, chosen), "train": ~np.isin(pool.day, chosen), "cut": None}
    if split == "chronological":
        days = np.array(sorted(pd.DatetimeIndex(list(data.span)).values.astype("datetime64[ns]")))
        cut = pd.Timestamp(days[-max(1, int(math.ceil(share * len(days))))])
        train = (pool.day < np.datetime64(cut.to_datetime64(), "ns")) & known_before(pool, cut)
        return {"held": test.day >= np.datetime64(cut.to_datetime64(), "ns"), "train": train, "cut": cut}
    raise ValueError(f"unknown design {split!r}; one of {SPLITS}")


def refit_forecasts(store: FitStore, anchor: pd.Timestamp, end: pd.Timestamp, every: int) -> np.ndarray | None:
    """Forecasts of every test origin by the model refitted every ``every`` days on the growing window (a failed
    refit keeps the model in use, as in the simulator); None if the fit at the anchor failed."""
    test_day = store.test_day
    fit_days = list(pd.date_range(anchor, end - pd.Timedelta(days=1), freq=f"{int(every)}D"))
    pred = None
    for i, fit_day in enumerate(fit_days):
        stop = fit_days[i + 1] if i + 1 < len(fit_days) else end
        rows = np.flatnonzero((test_day >= fit_day) & (test_day < stop))
        if not len(rows):
            continue
        window, used = store.serving(BASE, fit_days[: i + 1])
        part = store.forecast(window, used, rows)
        if part is None:
            return None
        if pred is None:
            pred = np.full((store.test.n, part.shape[1]), np.nan, dtype=np.float32)
        pred[rows] = part
    return pred


def fit_design(data: JobData, name: str, seed: int, settings: dict, device: str, train_mask: np.ndarray,
               held: np.ndarray) -> tuple[np.ndarray, dict]:
    """Fit the model on the training samples of a design and forecast its held-out origins."""
    train = data.pool.subset(train_mask)
    values, segment, step = data.series.between(data.base_start, data.end)
    fit = TrainData(train, values, segment, step,
                    validation=validation_mask(train.day, int(settings["validation_days"])))
    model = make_model(name, int(seed), settings, device)
    log = model.timed_fit(fit)
    begin = time.time()
    pred = model.predict(data.test.subset(held))
    return pred, {"train_samples": int(train.n), "train_days": int(len(np.unique(train.day))),
                  "fit_seconds": log.seconds, "compute_seconds": float(log.details.get("compute_seconds", log.seconds)),
                  "predict_seconds": round(time.time() - begin, 3)}


def evaluate_splits(job: dict) -> SplitResult:
    """The three designs and the simulator's frozen and refitted models on the same held-out origins of one patient,
    anchor and model (see the module docstring). ``job`` is a job of ``sim.cohort_runs`` with ``refit_every``
    (days), ``share`` (held-out share) and ``split_seed``."""
    started = time.time()
    patient_id, period, anchor_day, name = job["patient_id"], job["period"], int(job["anchor_day"]), job["model"]
    try:
        check_model(name)
        try:
            data = JobData(job)
        except LookupError as error:
            return SplitResult(patient_id, period, anchor_day, name, pd.DataFrame(), None, error=str(error))
        s, test, anchor, end = data.settings, data.test, data.anchor, data.end
        device = job.get("device", "cpu")
        clarke = ClarkeAccumulator(["model", "split", "arm", "horizon_min"], sample_size=int(s["clarke_point_sample"]),
                                   stream=int(job["stream"]))
        designs = {split: held_out(data, split, float(job["share"]), split_rng(job["split_seed"], patient_id, split))
                   for split in SPLITS}
        frames, info = [], []
        for seed in s["seeds"]:
            store = FitStore(name, seed, s, device, test, data.pool, data.series, data.base_start)
            frozen = store.forecast(BASE, anchor, np.arange(test.n))
            if frozen is None:  # no model at the anchor: the patient is left out, as in the simulator
                info.append({"seed": int(seed), "split": "", "excluded": "the fit at the anchor failed",
                             **store.get(BASE, anchor)["info"]})
                continue
            refitted = refit_forecasts(store, anchor, end, int(job["refit_every"]))
            for split, d in designs.items():
                held = d["held"]
                row = {"seed": int(seed), "split": split, "held_origins": int(held.sum()),
                       "held_days": int(len(np.unique(test.day[held]))),
                       "cut": d["cut"].date() if d["cut"] is not None else None}
                if not held.any():
                    info.append({**row, "error": "no held-out origins"})
                    continue
                try:
                    design, details = fit_design(data, name, seed, s, device, d["train"], held)
                    kept_days = np.unique(data.pool.day[d["train"]])
                    row.update(details, train_days_after_first_held=int((kept_days > test.day[held].min()).sum()))
                except Exception as error:  # noqa: BLE001
                    row["error"] = f"{type(error).__name__}: {str(error).strip()}"[:300]
                    info.append(row)
                    continue
                info.append(row)
                arms = {"design": design, "frozen": frozen[held], "persistence": data.reference[held]}
                if refitted is not None:
                    arms["refitted"] = refitted[held]
                part = test.subset(held)
                for arm, pred in arms.items():
                    table = day_metrics(pred, part.targets, part.day, part.horizons_min)
                    if table.empty:
                        continue
                    for column, value in (("arm", arm), ("split", split), ("seed", int(seed)), ("model", name)):
                        table.insert(0, column, value)
                    frames.append(table)
                    for k, h in enumerate(part.horizons_min):
                        clarke.add((name, split, arm, h), part.targets[:, k], pred[:, k])
            info.append({"seed": int(seed), "split": "simulator", "refit_every": int(job["refit_every"]),
                         "fits": len(store.fits), "fits_failed": sum(e["pred"] is None for e in store.fits.values()),
                         "compute_seconds": round(sum(float(e["info"].get("compute_seconds", 0) or 0)
                                                      for e in store.fits.values()), 3)})
            del store
        days = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if not days.empty:
            for column, value in (("anchor_day", anchor_day), ("period", period), ("patient_id", patient_id)):
                days.insert(0, column, value)
        info = [{"patient_id": patient_id, "period": period, "anchor_day": anchor_day, "model": name, **r}
                for r in info]
        return SplitResult(patient_id, period, anchor_day, name, days, clarke, info, round(time.time() - started, 2))
    except Exception:  # noqa: BLE001
        return SplitResult(patient_id, period, anchor_day, name, pd.DataFrame(), None,
                           seconds=round(time.time() - started, 2), error=traceback.format_exc(limit=6))
