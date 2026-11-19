"""Stage 0 for one patient (scripts 10 and 12): frozen models on the patient's follow-up.

For each period (whole follow-up, after randomisation) the patient's day 0
comes from script 04. The training window is the ``warmup_days`` days from
day 0 (frozen models); every later day of the period is a test day. Origins
follow ``sim.samples`` with the cohort's slot conditions (open loop, insulin
known), so all models see the same origins. Training samples whose targets
reach the first test day are dropped (HANDOFF 7.1). Learned models hold out
the samples of the last ``validation_days`` days of the window that have
samples (D62) for early stopping.

Learned models come in three starting points (D62): ``lightgbm`` / ``nhits``
(scratch, the patient's window only), ``*_pop`` (pretrained on the pool group
by script 11, used as is) and ``*_ft`` (pretrained, then fine-tuned on the
window). Pretrained models are read from ``settings["population_dir"]``.

Returned per (period, model, horizon, test day): number of forecasts, sums of
squared, absolute and relative errors, sum of errors and Clarke zone counts,
plus a ``ClarkeAccumulator`` with the grid of every (model, horizon, period).
"""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.store import ProcessedStore
from ..eval.metrics import ZONES, ClarkeAccumulator, clarke_zone_codes
from ..models.arx import RecursiveARX
from ..models.base import EXOG, TrainData
from ..models.foundation import COVARIATES, DEFAULT_CONTEXT_HOURS, FOUNDATION_MODELS, FoundationModel
from ..models.gbdt import LightGBMDirect
from ..models.neural import NETWORKS, NeuralForecastModel
from ..models.reference import REFERENCE_MODELS
from .samples import known_before, make_samples, training_series

ORIGIN_RULES = {  # slot conditions of the origins of each usable-day rule of the cohorts (data.splits.USABLE_RULES)
    "open_insulin": ("valid_day", "open_loop", "insulin_known"),
    "open": ("valid_day", "open_loop"),
    "closed_insulin": ("valid_day", "closed_loop", "insulin_known"),
    "valid": ("valid_day",),
    "insulin": ("valid_day", "insulin_known"),
}
ORIGIN_CONDITIONS = ORIGIN_RULES["open_insulin"]  # the core cohort's usable days (D50)
FAMILIES = ("lightgbm",) + tuple(NETWORKS)  # learned model families (scratch, *_pop, *_ft)
STATISTICAL_MODELS = {"arx": RecursiveARX}  # fitted per patient, deterministic (no seeds, no population)
CONTEXT_MODELS = tuple(FOUNDATION_MODELS)  # zero-shot foundation models: no fit, a long history per forecast
STARTS = {"": "scratch", "_pop": "population", "_ft": "finetune"}
TWIN = "noexog"  # the glucose-only twin of a model (D76, D90): no IOB, COB, insulin activity, absorption or hour
VARIANTS = {f"_{TWIN}": TWIN}  # name suffix -> variant: the same model with other inputs (only the twin so far)


def split_variant(name: str) -> tuple[str, str]:
    """('lightgbm_ft', 'noexog') for 'lightgbm_ft_noexog'; ('lightgbm_ft', '') for 'lightgbm_ft'."""
    for suffix, variant in VARIANTS.items():
        if name.endswith(suffix):
            return name[: -len(suffix)], variant
    return name, ""


def add_loop_columns(grid: pd.DataFrame) -> pd.DataFrame:
    """The slot conditions ``open_loop`` and ``closed_loop`` (the loop mode of the slot's day, script 03)."""
    loop = grid["loop"]
    grid["open_loop"] = loop.eq("open").fillna(False).astype(bool)
    grid["closed_loop"] = loop.eq("closed").fillna(False).astype(bool)
    return grid


def is_twin(name: str) -> bool:
    """Whether ``name`` is a glucose-only twin (D90)."""
    return split_variant(name)[1] == TWIN


def split_name(name: str) -> tuple[str, str]:
    """('nhits', 'finetune') for 'nhits_ft' (and for 'nhits_ft_noexog'); reference models have start ''."""
    name = split_variant(name)[0]
    for suffix, start in STARTS.items():
        if suffix and name.endswith(suffix) and name[: -len(suffix)] in FAMILIES:
            return name[: -len(suffix)], start
    return name, "scratch" if name in FAMILIES else ""


def is_learned(name: str) -> bool:
    return split_name(name)[0] in FAMILIES


def population_path(folder: str | Path, family: str, seed: int, variant: str = "") -> Path:
    """Where script 11 saves the pretrained model of a family and seed (``variant``: the glucose-only twin, D90)."""
    folder = Path(folder)
    stem = f"{family}_{variant}_seed{seed}" if variant else f"{family}_seed{seed}"
    return folder / (f"{stem}.pt" if family in NETWORKS else stem)


def mechanism_of(name: str) -> str:
    """How a model takes in new data (HANDOFF 6.1): re_estimation, sample_wise, continue_training, global_finetune,
    context_only or reference."""
    base = split_variant(name)[0]
    if base in REFERENCE_MODELS:
        return REFERENCE_MODELS[base].mechanism
    if base in STATISTICAL_MODELS:
        return STATISTICAL_MODELS[base].mechanism
    if base in CONTEXT_MODELS:
        return FoundationModel.mechanism
    family = split_name(base)[0]
    if family in NETWORKS:
        return NeuralForecastModel.mechanism
    return LightGBMDirect.mechanism if family == "lightgbm" else "unknown"


def needs_exog(name: str) -> bool:
    """Whether the samples of a model carry IOB, COB and the time of day (learned models and ARX)."""
    return is_learned(name) or split_variant(name)[0] in STATISTICAL_MODELS


# ----------------------------------------------------------------------------- glucose-only twins (D90)
#
# The inputs are a factor of the whole study: every model that reads insulin and carbohydrates and goes on to the
# later stages (the fine-tuned learned models of ``glucose_only_twins.families`` and ARX) has a twin that reads the
# glucose only, and the twin goes through exactly what its full model goes through (versions, policies, arms,
# intervals). Selections are made on the full models and the twins inherit them, so the effect of insulin and
# carbohydrates is always a paired comparison. AR(3), PatchTST and the foundation models read the glucose only.


def twin_of(name: str, twin_families) -> str | None:
    """The glucose-only twin of ``name`` (D90), or None: ARX, and the fine-tuned learned models of
    ``twin_families`` (``glucose_only_twins.families``); no twins at all when ``twin_families`` is None."""
    base, variant = split_variant(name)
    if variant or twin_families is None:
        return None
    family, start = split_name(base)
    if base in STATISTICAL_MODELS or (family in set(twin_families) and start == "finetune"):
        return f"{base}_{TWIN}"
    return None


def with_twins(models, twin_families) -> list[str]:
    """``models`` with the glucose-only twin of each right after it (D90); a twin listed already is kept once."""
    out = []
    for model in models:
        for name in (model, twin_of(model, twin_families)):
            if name and name not in out:
                out.append(name)
    return out


def twin_available(name: str, population_dir, seeds) -> bool:
    """Whether a model can run here: a glucose-only learned twin needs its population models of script 11 for
    every seed; every other model is taken as available."""
    base, variant = split_variant(name)
    family, start = split_name(base)
    if not variant or family not in FAMILIES or start == "scratch":
        return True
    return all(population_path(population_dir, family, int(s), variant).exists() for s in seeds)


def twin_families(config: dict) -> list[str] | None:
    """The learned families with glucose-only twins (``glucose_only_twins.families``; ARX always has one), or None
    (no twins at all) when ``glucose_only_twins.enabled`` is false."""
    twins = config.get("glucose_only_twins") or {}
    if not twins.get("enabled", True):
        return None
    return [str(f) for f in twins.get("families", [])]


def model_dependencies(name: str, settings: dict, choices: dict) -> dict:
    """What a model's results depend on besides the data and the stage settings, for the result caches of scripts
    20, 30, 40, 42, 43, 50 and 71-73: the settings of its family, statistical model or context model (its context,
    D99) and, for a learned model, its entry in ``choices.yaml`` of script 11 (with the date of the run; a
    glucose-only twin adds its own population models)."""
    base, variant = split_variant(name)
    family = split_name(base)[0]
    learned = family in FAMILIES
    has_settings = learned or family in STATISTICAL_MODELS or base in CONTEXT_MODELS
    out = {"model_settings": settings.get(family) if has_settings else None, "choices": None}
    if learned:
        entry = choices.get(family) or {}
        own = {k: v for k, v in entry.items() if k not in VARIANTS.values()}
        own.setdefault("created", choices.get("created"))
        if variant:
            own[variant] = entry.get(variant)
        out["choices"] = own
    return out


def per_model(table: dict, name: str, default=1):
    """A per-model setting (e.g. ``stage1.refresh_days``): a twin takes that of its full model (D90)."""
    return table.get(name, table.get(split_variant(name)[0], default))


def model_options(settings: dict) -> dict:
    """The model options of the later analyses (``sim.cohort_runs``), for LightGBM and ARX only: ``low_loss``
    (script 71, D96) and the extra inputs ``extra_inputs`` (script 73, D98); empty when the settings set none."""
    options = {}
    if settings.get("low_loss"):
        options["low_loss"] = dict(settings["low_loss"])
    if settings.get("extra_inputs"):
        options["extra"] = tuple(settings["extra_inputs"])
    return options


def input_columns(name: str, settings: dict) -> tuple[str, ...]:
    """The grid columns the samples of a model carry besides glucose: IOB, COB, activity, absorption and the time of
    day for the learned models and ARX, plus the extra inputs of the settings (D98)."""
    return EXOG + tuple(settings.get("extra_inputs") or ()) if needs_exog(name) else ()


def make_model(name: str, seed: int, settings: dict, device: str):
    if name in REFERENCE_MODELS:
        return REFERENCE_MODELS[name]()
    base, variant = split_variant(name)
    options = model_options(settings)
    if base in STATISTICAL_MODELS:
        return STATISTICAL_MODELS[base](**(settings.get(base) or {}), **({"exog": False} if variant == TWIN else {}),
                                        **options)
    if name in CONTEXT_MODELS:
        own = settings.get(name) or {}
        return FoundationModel(name, context_hours=context_hours(name, settings), device=device,
                               batch_size=int(own.get("batch_size", settings.get("foundation_batch_size", 256))),
                               model_id=own.get("model_id"), revision=own.get("revision"))
    family, start = split_name(name)
    if family not in FAMILIES:
        raise ValueError(f"unknown model {name!r}")
    population = population_path(settings["population_dir"], family, seed, variant) if start != "scratch" else None
    finetune = settings.get("finetune", {}).get(family)
    params = settings.get(family)
    if variant == TWIN:  # a pretrained model brings its own setting; a scratch model takes it from here
        params = {**(params or {}), "exog": False}
    if family in NETWORKS:
        if options:
            raise ValueError(f"{name}: the options {sorted(options)} exist for LightGBM and ARX only")
        return NeuralForecastModel(family, seed, params, device, start, population, finetune)
    return LightGBMDirect(seed, params, start, population, finetune, **options)


def context_hours(name: str, settings: dict) -> float:
    own = settings.get(name) or {}
    return float(own.get("context_hours", settings.get("context_hours", DEFAULT_CONTEXT_HOURS)))


def context_for(models: list[str], settings: dict) -> dict:
    """make_samples arguments for the long history of the context-only models among ``models`` (if any)."""
    hours = [context_hours(m, settings) for m in models if m in CONTEXT_MODELS]
    return {"context_h": max(hours), "context_exog": COVARIATES} if hours else {}


def validation_mask(day: np.ndarray, validation_days: int) -> np.ndarray:
    """Samples of the last ``validation_days`` days that have samples (D62)."""
    if not len(day) or validation_days <= 0:
        return np.zeros(len(day), dtype=bool)
    days = np.unique(day)
    held = days[-validation_days:] if len(days) > validation_days else days[:0]
    return np.isin(day, held)


@dataclass
class Stage0Result:
    patient_id: str
    days: pd.DataFrame
    clarke: ClarkeAccumulator | None
    info: list = field(default_factory=list)
    seconds: float = 0.0
    error: str = ""
    warnings: str = ""


def day_metrics(pred: np.ndarray, target: np.ndarray, days: np.ndarray, horizons: list[int],
                thresholds: tuple = ()) -> pd.DataFrame:
    """Per horizon and day: number of forecasts, sums of squared, absolute, relative and signed errors, and Clarke
    zone counts. For every glucose threshold ``t`` of ``thresholds`` (the confirmation: 70 and 54 mg/dL), also the
    forecasts whose reference is below it (``below<t>_true``), those that forecast it (``below<t>_alarm``) and
    those that do both (``below<t>_hit``), for the sensitivity and precision of the forecast low."""
    rows = []
    for k, h in enumerate(horizons):
        ok = np.isfinite(target[:, k]) & np.isfinite(pred[:, k])
        if not ok.any():
            continue
        r, p, d = target[ok, k].astype(float), pred[ok, k].astype(float), days[ok]
        e = p - r
        frame = pd.DataFrame({"day": d, "n": 1, "sse": e ** 2, "sae": np.abs(e), "sape": np.abs(e) / r, "se": e})
        zone = clarke_zone_codes(r, p)
        for z, letter in enumerate(ZONES):
            frame[f"zone_{letter}"] = (zone == z).astype(int)
        for threshold in thresholds:
            low_true, low_alarm = r < threshold, p < threshold
            frame[f"below{threshold:g}_true"] = low_true.astype(int)
            frame[f"below{threshold:g}_alarm"] = low_alarm.astype(int)
            frame[f"below{threshold:g}_hit"] = (low_true & low_alarm).astype(int)
        grouped = frame.groupby("day").sum().reset_index()
        grouped.insert(0, "horizon_min", h)
        rows.append(grouped)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def evaluate_patient(job: dict) -> Stage0Result:
    started = time.time()
    patient_id = job["patient_id"]
    try:
        grid = ProcessedStore(job["dataset"], root=job["derived_root"], stage="features").read_frame("grid", patient_id)
        if grid.empty:
            return Stage0Result(patient_id, pd.DataFrame(), None, error="no feature grid")
        add_loop_columns(grid)
        grid_days = pd.DatetimeIndex(grid["day"].unique())
        period_days = {"whole": set(grid_days),
                       "post_randomization": set(pd.DatetimeIndex(grid.loc[grid["period"] == "post_randomization",
                                                                           "day"].unique()))}
        settings = job["settings"]
        clarke = ClarkeAccumulator(["model", "seed", "horizon_min", "period"],
                                   sample_size=int(settings["clarke_point_sample"]), stream=int(job["stream"]))
        learned = [m for m in settings["models"] if is_learned(m)]
        exog = EXOG if any(needs_exog(m) for m in settings["models"]) else ()
        frames, info = [], []
        for period, day0 in job["day0"].items():
            if day0 is None or pd.isna(day0):
                continue
            day0 = pd.Timestamp(day0)
            warmup_end = day0 + pd.Timedelta(days=int(settings["warmup_days"]))
            train_days = {d for d in period_days[period] if day0 <= d < warmup_end}
            test_days = {d for d in period_days[period] if d >= warmup_end}
            common = dict(horizons_min=settings["horizons_min"], lookback_h=settings["lookback_hours"],
                          min_window_share=settings["min_window_share"],
                          recent_complete_min=settings["recent_complete_min"], origin_conditions=ORIGIN_CONDITIONS)
            test = make_samples(grid, stride_min=settings["origin_stride_min"], allowed_days=test_days,
                                exog_columns=exog, **common, **context_for(settings["models"], settings))
            train = make_samples(grid, stride_min=settings["train_stride_min"], allowed_days=train_days,
                                 exog_columns=exog, trajectory=bool(learned), **common)
            train = train.subset(known_before(train, warmup_end))  # no target on a test day (7.1)
            series, segment, step = training_series(grid, train_days)
            held = validation_mask(train.day, int(settings.get("validation_days", 2)))
            data = TrainData(train, series, segment, step, validation=held)
            for name in settings["models"]:
                for seed in (settings["seeds"] if is_learned(name) else [0]):
                    base = {"patient_id": patient_id, "period": period, "model": name, "seed": int(seed),
                            "day0": day0.date(), "train_samples": train.n,
                            "validation_samples": int(data.validation.sum()), "test_origins": test.n}
                    try:  # a model that cannot be fitted (e.g. no training sample) does not stop the others
                        model = make_model(name, int(seed), settings, job.get("device", "cpu"))
                        log = model.timed_fit(data)
                        started_predict = time.time()
                        pred = model.predict(test)
                    except Exception as error:  # noqa: BLE001
                        info.append({**base, "error": f"{type(error).__name__}: {str(error).strip()}"[:300]})
                        continue
                    table = day_metrics(pred, test.targets, test.day, test.horizons_min)
                    if not table.empty:
                        table.insert(0, "seed", int(seed))
                        table.insert(0, "model", name)
                        table.insert(0, "period", period)
                        frames.append(table)
                    for k, h in enumerate(test.horizons_min):
                        clarke.add((name, int(seed), h, period), test.targets[:, k], pred[:, k])
                    info.append({**base, "fit_samples": log.n_samples, "fit_seconds": log.seconds,
                                 "predict_seconds": round(time.time() - started_predict, 3),
                                 "step_min": test.step_min, **log.details})
        days = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if not days.empty:
            days.insert(0, "patient_id", patient_id)
            starts = {p: pd.Timestamp(d) + pd.Timedelta(days=int(job["settings"]["warmup_days"]))
                      for p, d in job["day0"].items() if d is not None and not pd.isna(d)}
            days["age_days"] = (pd.to_datetime(days["day"]) - days["period"].map(starts)).dt.days
        return Stage0Result(patient_id, days, clarke, info, round(time.time() - started, 2))
    except Exception:  # noqa: BLE001
        return Stage0Result(patient_id, pd.DataFrame(), None, seconds=round(time.time() - started, 2),
                            error=traceback.format_exc(limit=6))
