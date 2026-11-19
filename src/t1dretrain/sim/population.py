"""Samples of the pool group for population models (script 11; D62).

The pool group is never evaluated (D12), so every usable day of a pool patient
may be used to pretrain. For each pool patient this module builds:

- ``population``: samples on every usable day of the whole follow-up, with the
  cohort's origin conditions (valid day, open loop, insulin known), on a
  coarser stride than the stage-0 training samples (windows overlap anyway);
- ``warmup``: the stage-0 training window (14 days from the whole-period day
  0, every slot) with its validation days, used to fine-tune and to stop the
  population training early on the check patients;
- ``test``: the stage-0 test origins (every 30 min on the later days), used to
  score hyperparameters on the check patients.

The pool is split once into *fit* patients, whose population samples train
the candidate population models, and *check* patients, on which the
candidates are scored as a new patient would be: zero-shot and fine-tuned on
their first 14 days, tested on their later days. The final population models
are trained on every pool patient.
"""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..data.store import ProcessedStore
from ..models.base import EXOG
from .samples import Samples, known_before, make_samples, training_series
from .stage0 import ORIGIN_CONDITIONS, add_loop_columns, validation_mask


@dataclass
class PoolPatient:
    patient_id: str
    population: Samples | None = None
    warmup: Samples | None = None
    warmup_validation: np.ndarray | None = None
    warmup_series: tuple | None = None  # (glucose on contiguous slots, segment, step) of the window (AR models)
    test: Samples | None = None
    info: dict | None = None
    seconds: float = 0.0
    error: str = ""


def split_pool(patients: list[str], check_share: float, seed: int) -> tuple[list[str], list[str]]:
    """(fit, check) patients: a fixed random share of the pool held out to score hyperparameters."""
    ordered = sorted(patients)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(ordered))
    n_check = max(1, int(round(check_share * len(ordered)))) if len(ordered) > 1 else 0
    check = sorted(ordered[i] for i in order[:n_check])
    return [p for p in ordered if p not in set(check)], check


def pool_patient(job: dict) -> PoolPatient:
    """Samples of one pool patient (see the module docstring)."""
    started = time.time()
    patient_id = job["patient_id"]
    try:
        grid = ProcessedStore(job["dataset"], root=job["derived_root"], stage="features").read_frame("grid", patient_id)
        if grid.empty:
            return PoolPatient(patient_id, error="no feature grid")
        add_loop_columns(grid)
        s = job["settings"]
        common = dict(horizons_min=s["horizons_min"], lookback_h=s["lookback_hours"],
                      min_window_share=s["min_window_share"], recent_complete_min=s["recent_complete_min"],
                      origin_conditions=ORIGIN_CONDITIONS, exog_columns=EXOG)
        population = make_samples(grid, stride_min=s["population_stride_min"], trajectory=True, **common)
        result = PoolPatient(patient_id, population=population)
        info = {"patient_id": patient_id, "population_samples": population.n,
                "usable_days": int(len(np.unique(population.day)))}
        day0 = job.get("day0")
        if day0 is not None and not pd.isna(day0):
            day0 = pd.Timestamp(day0)
            warmup_end = day0 + pd.Timedelta(days=int(s["warmup_days"]))
            days = pd.DatetimeIndex(grid["day"].unique())
            train_days = set(days[(days >= day0) & (days < warmup_end)])
            test_days = set(days[days >= warmup_end])
            result.warmup = make_samples(grid, stride_min=s["train_stride_min"], allowed_days=train_days,
                                         trajectory=True, **common)
            result.warmup = result.warmup.subset(known_before(result.warmup, warmup_end))  # no target on a test day
            result.warmup_validation = validation_mask(result.warmup.day, int(s["validation_days"]))
            result.warmup_series = training_series(grid, train_days)
            result.test = make_samples(grid, stride_min=s["origin_stride_min"], allowed_days=test_days, **common)
            info.update({"day0": day0.date(), "warmup_samples": result.warmup.n,
                         "validation_samples": int(result.warmup_validation.sum()), "test_origins": result.test.n})
        result.info = info
        result.seconds = round(time.time() - started, 2)
        return result
    except Exception:  # noqa: BLE001
        return PoolPatient(patient_id, seconds=round(time.time() - started, 2), error=traceback.format_exc(limit=6))


@dataclass
class ScoreResult:
    patient_id: str
    days: pd.DataFrame
    clarke: object = None
    info: list | None = None
    seconds: float = 0.0
    error: str = ""
    warnings: str = ""


def build_variant(variant: dict, device: str):
    """A model from a variant description: {family, start, params, population, finetune, seed}."""
    from ..models.gbdt import LightGBMDirect
    from ..models.neural import NETWORKS, NeuralForecastModel
    from ..models.reference import REFERENCE_MODELS

    family = variant["family"]
    if family in REFERENCE_MODELS:
        return REFERENCE_MODELS[family]()
    seed = int(variant.get("seed", 1))
    if family in NETWORKS:
        return NeuralForecastModel(family, seed, variant.get("params"), device, variant.get("start", "scratch"),
                                   variant.get("population"), variant.get("finetune"))
    if family == "lightgbm":
        return LightGBMDirect(seed, variant.get("params"), variant.get("start", "scratch"), variant.get("population"),
                              variant.get("finetune"))
    raise ValueError(f"unknown model family {family!r}")


def score_patient(job: dict) -> ScoreResult:
    """Fit every variant on a pool patient's 14-day window and score it on the later days (stage-0 protocol).

    ``job``: {"patient": PoolPatient with warmup, validation and test samples, "variants": [{"key", "family",
    "start", "params", "population", "finetune", "seed"}], "device", "clarke_point_sample", "stream"}.
    """
    from ..eval.metrics import ClarkeAccumulator
    from ..models.base import TrainData
    from .stage0 import day_metrics

    started = time.time()
    patient: PoolPatient = job["patient"]
    try:
        clarke = ClarkeAccumulator(["model", "seed", "horizon_min", "period"],
                                   sample_size=int(job["clarke_point_sample"]), stream=int(job["stream"]))
        train, test = patient.warmup, patient.test
        data = TrainData(train, *patient.warmup_series, validation=patient.warmup_validation)
        frames, info = [], []
        for variant in job["variants"]:
            seed = int(variant.get("seed", 0))
            try:  # a variant that cannot be fitted does not stop the others
                model = build_variant(variant, job.get("device", "cpu"))
                log = model.timed_fit(data)
                begin = time.time()
                pred = model.predict(test)
            except Exception as error:  # noqa: BLE001
                info.append({"patient_id": patient.patient_id, "model": variant["key"], "seed": seed,
                             "train_samples": train.n, "test_origins": test.n,
                             "error": f"{type(error).__name__}: {str(error).strip()}"[:300]})
                continue
            table = day_metrics(pred, test.targets, test.day, test.horizons_min)
            if not table.empty:
                table.insert(0, "seed", seed)
                table.insert(0, "model", variant["key"])
                table.insert(0, "period", "whole")
                frames.append(table)
            for k, h in enumerate(test.horizons_min):
                clarke.add((variant["key"], seed, h, "whole"), test.targets[:, k], pred[:, k])
            details = {k: (str(v) if isinstance(v, (list, dict)) else v) for k, v in log.details.items()
                       if k != "curve"}
            info.append({"patient_id": patient.patient_id, "model": variant["key"], "seed": seed,
                         "train_samples": train.n, "fit_samples": log.n_samples, "fit_seconds": log.seconds,
                         "predict_seconds": round(time.time() - begin, 3), "test_origins": test.n, **details})
        days = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if not days.empty:
            days.insert(0, "patient_id", patient.patient_id)
        return ScoreResult(patient.patient_id, days, clarke, info, round(time.time() - started, 2))
    except Exception:  # noqa: BLE001
        return ScoreResult(patient.patient_id, pd.DataFrame(), None, seconds=round(time.time() - started, 2),
                           error=traceback.format_exc(limit=6))


def train_population(job: dict) -> dict:
    """Train one population model on cached pool samples and save it (script 11).

    ``job``: {"family": "lightgbm" | "nhits", "params", "seed", "cache" (folder written by ``save_samples``, with
    ``validation.npy`` when samples are held out), "out" (where the model is saved), "device", "threads", "key",
    "low_loss" (LightGBM only, script 71: the loss that weighs the lows, D96)}.
    LightGBM stops early on the held-out samples (else ``max_rounds``, an int or one per horizon); N-HiTS too
    (else ``max_steps``). Returns a log with sizes, seconds, rounds or steps and the validation curve.
    """
    from pathlib import Path

    from .samples import load_samples

    started = time.time()
    log = {"key": job.get("key"), "family": job["family"], "seed": int(job["seed"]), "status": "ok"}
    try:
        samples = load_samples(job["cache"])
        held_file = Path(job["cache"]) / "validation.npy"
        validation = np.load(held_file) if held_file.exists() else None
        log.update({"samples": int(samples.n), "validation_samples": int(validation.sum()) if validation is not None
                    else 0, "patients": int(len(np.unique(samples.patient))) if samples.patient is not None else None})
        out = Path(job["out"])
        out.parent.mkdir(parents=True, exist_ok=True)
        if job["family"] == "lightgbm":
            from ..models.gbdt import LightGBMDirect

            model = LightGBMDirect(int(job["seed"]), job["params"], threads=int(job.get("threads", 1)),
                                   low_loss=job.get("low_loss"))
            fit = model.train_on(samples, validation, model.params)
            model.save(out)
            log.update({"rounds": fit.details["rounds"]})
            if validation is not None and validation.any():
                held = samples.subset(validation)
                pred = model.predict(held)
                log["validation_rmse"] = [round(float(np.sqrt(np.nanmean((pred[:, k] - held.targets[:, k]) ** 2))), 3)
                                          for k in range(pred.shape[1])]
        else:
            from ..models.neural import SCHEDULE, NeuralForecastModel

            model = NeuralForecastModel(job["family"], int(job["seed"]), job["params"], job.get("device", "cpu"))
            fit = model.train_on(samples, validation, {k: model.params[k] for k in SCHEDULE})
            model.save(out)
            log.update({k: fit.details.get(k) for k in ("steps", "best_step", "best_val_mse", "device", "cuda_error")})
            log["curve"] = fit.details.get("curve", [])
        log["fit_samples"] = int(fit.n_samples)
    except Exception:  # noqa: BLE001
        log.update({"status": "error", "error": traceback.format_exc(limit=6)})
    log["seconds"] = round(time.time() - started, 1)
    return log


def expand_grid(base: dict, grid: dict | None) -> list[dict]:
    """Every combination of the ``grid`` values on top of ``base`` (keys in sorted order; ``base`` alone if empty)."""
    import itertools

    grid = grid or {}
    keys = sorted(grid)
    values = [grid[k] if isinstance(grid[k], (list, tuple)) else [grid[k]] for k in keys]
    return [{**base, **dict(zip(keys, combo))} for combo in itertools.product(*values)] or [dict(base)]
