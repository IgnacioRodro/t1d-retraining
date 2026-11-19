"""Jobs of the confirmation machinery (script 50, and scripts 43, 71 to 73 and 75 that run on it; HANDOFF 7.6, D92).

One job per cohort, patient, anchor and model runs every arm of that model
(``sim.mechanisms.evaluate_mechanisms``) and caches each arm, and the
persistence of the patient, on its own (``sim.cache``), so that a run that
stops can go on later. What runs comes from the screening: the policies of
script 31 (``confirmation.yaml``, D84) and the mechanism of script 41
(``mechanisms.yaml``, D91).

A script on this machinery may change what its models learn through two model
options of the settings, which enter the cache key only when they are set (so
the keys of script 50 do not change):

- ``low_loss`` ({"threshold", "factor"}; script 71, D96): training targets
  below the threshold weigh ``factor`` times in the loss (LightGBM and ARX);
- ``extra_inputs`` (grid columns; script 73, D98): more inputs read at the
  forecast origin (the wearables; LightGBM and ARX).

Script 43 (D99) runs the zero-shot foundation models on it with their own
context length (``context_hours`` in their settings, part of the cache key).
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ..data.pipeline import WarningRecorder
from ..data.splits import group_patients
from ..data.store import ProcessedStore
from ..eval.confirmation import cohort_key, draw_patients
from ..eval.metrics import ClarkeAccumulator
from .cache import cache_valid, read_cache, retryable, write_cache
from .stage0 import FAMILIES, STATISTICAL_MODELS, model_dependencies, population_path, split_name, split_variant

CORE_STEP_MIN = 5.0  # the grid of the core dataset, which the population models read
SETTING_KEYS = ("horizons_min", "lookback_hours", "origin_stride_min", "train_stride_min", "validation_days",
                "min_window_share", "recent_complete_min", "window_days", "test_days", "seeds", "event_thresholds")
OPTION_KEYS = ("low_loss", "extra_inputs")  # model options (scripts 71 and 73): in the cache key only when set
FIT_LOG_COLUMNS = ("policy", "test_origins", "train_days", "train_samples", "fit_seconds", "compute_seconds",
                   "rounds", "added_rounds", "steps", "best_step", "error", "excluded", "served_by", "skipped",
                   "p_value")


# ============================================================================ settings and plan


def base_settings(config: dict, choices: dict, population_dir: Path) -> dict:
    """The settings of every confirmation job (``test_days`` is the cohort's): stage-0 protocol, origins every
    ``origin_stride_min_confirm`` minutes, seeds and thresholds of the forecast lows of ``confirmation``, the
    population models of script 11 and their fine-tuning, the model settings of stage 0."""
    stage, conf = config["stage0"], config["confirmation"]
    base = {"horizons_min": [int(h) for h in config["horizons_min"]], "lookback_hours": float(config["lookback_hours"]),
            "origin_stride_min": float(config["origin_stride_min_confirm"]),
            "train_stride_min": float(stage["train_stride_min"]), "validation_days": int(stage["validation_days"]),
            "min_window_share": float(stage["min_window_share"]),
            "recent_complete_min": float(stage["recent_complete_min"]),
            "window_days": int(config["anchors"]["train_window_days"]),
            "clarke_point_sample": int(config["evaluation"]["clarke_point_sample"]),
            "seeds": [int(s) for s in conf["seeds"]], "event_thresholds": [float(t) for t in conf["event_thresholds"]],
            "population_dir": str(population_dir),
            "finetune": {f: (choices.get(f) or {}).get("finetune") for f in FAMILIES}}
    for f in FAMILIES:
        base[f] = (choices.get(f) or {}).get("scratch") or stage.get(f)
    for m in STATISTICAL_MODELS:
        base[m] = stage.get(m) or {}
    return base


def read_choices(population_dir: Path) -> dict:
    path = population_dir / "choices.yaml"
    if not path.exists():
        raise SystemExit(f"{path} not found: run scripts/11_pretrain_population.py first")
    return yaml.safe_load(path.read_text()) or {}


def read_plan(derived: Path) -> tuple[dict, dict]:
    """``confirmation.yaml`` of script 31 and ``mechanisms.yaml`` of script 41 (both required)."""
    out = []
    for script, name in (("31_stage2_policy_analysis", "confirmation.yaml"),
                         ("41_stage3_mechanism_analysis", "mechanisms.yaml")):
        path = derived / "runs" / script / name
        if not path.exists():
            raise SystemExit(f"{path} not found: run scripts/{script}.py first")
        out.append(yaml.safe_load(path.read_text()) or {})
    return out[0], out[1]


def plan_models(policies_doc: dict, mechanisms_doc: dict, chosen: list[str] | None, population_dir: Path,
                seeds: list[int]) -> tuple[list[str], dict, list[str]]:
    """The models of script 31 with their twins, and per model its policies, optimal interval and mechanism (a
    twin takes its full model's). ``chosen`` filters by the full model's name; learned models need their
    population models of script 11."""
    models, plan, notes = [], {}, []
    mechanisms = mechanisms_doc.get("models") or {}
    for model, entry in (policies_doc.get("models") or {}).items():
        if chosen and model not in chosen:
            continue
        mechanism = (mechanisms.get(model) or {}).get("arm")
        if split_name(model)[0] in FAMILIES and mechanism is None:
            notes.append(f"{model}: no mechanism in mechanisms.yaml (script 41): the restart only")
        for name in [model] + ([entry["twin"]] if entry.get("twin") else []):
            family, start = split_name(name)
            if family in FAMILIES and start != "scratch":
                missing = [s for s in seeds
                           if not population_path(population_dir, family, int(s), split_variant(name)[1]).exists()]
                if missing:
                    notes.append(f"{name} left out: no population model for seed(s) {missing}")
                    continue
            models.append(name)
            plan[name] = {"policies": list(entry.get("policies") or ["never"]),
                          "interval": entry.get("optimal_interval_days"), "mechanism": mechanism}
    return models, plan, notes


# ============================================================================ patients


def grid_steps(dataset: str, derived: Path) -> pd.Series:
    """The grid step (minutes) of every patient of a dataset (script 05)."""
    features = ProcessedStore(dataset, root=derived, stage="features").read_extra("feature_patients")
    return features.set_index("patient_id")["step_min"] if not features.empty else pd.Series(dtype=float)


def on_core_grid(patients, steps: pd.Series) -> list[str]:
    """The patients whose grid has the core dataset's step (the learned models read 5-min inputs)."""
    return [p for p in patients if float(steps.get(p, np.nan)) == CORE_STEP_MIN]


def cohort_patients(c: dict, derived: Path, anchor_days: list[int], seed: int,
                    first: int | None) -> tuple[pd.DataFrame, dict]:
    """The anchors of one cohort's patients (available at the anchor days, in the cohort's group if it names one,
    on a 5-min grid, drawn if the cohort asks for fewer) and the counts behind them."""
    dataset, cohort, period = c["dataset"], c["cohort"], c["period"]
    anchors = ProcessedStore(dataset, root=derived, stage="clean").read_extra("anchors")
    anchors = anchors[(anchors["cohort"] == cohort) & (anchors["period"] == period) &
                      anchors["available"].astype(bool) & anchors["anchor_day"].isin(anchor_days)].copy()
    if c.get("group"):
        anchors = anchors[anchors["patient_id"].isin(group_patients(dataset, c["group"], allow_confirm=True))]
    counts = {"with_anchor": int(anchors["patient_id"].nunique())}
    on_grid = on_core_grid(anchors["patient_id"].unique(), grid_steps(dataset, derived))
    counts["left_out_grid"] = counts["with_anchor"] - len(on_grid)
    drawn = draw_patients(on_grid, c.get("patients"), seed, cohort_key(dataset, cohort))
    if first:
        drawn = drawn[:first]
    counts["patients"] = len(drawn)
    anchors = anchors[anchors["patient_id"].isin(drawn)].copy()
    anchors["anchor"] = pd.to_datetime(anchors["day0"]) + pd.to_timedelta(anchors["anchor_day"], unit="D")
    return anchors.sort_values(["patient_id", "anchor_day"]).reset_index(drop=True), counts


# ============================================================================ jobs and cache


def cache_key(cohort: str, rule: str, model: str, arm: str, settings: dict, choices: dict) -> str:
    """What a cached result depends on: cohort and its usable rule, model, arm, starting point, settings, model
    options (when set) and simulator."""
    from .mechanisms import CACHE_VERSION

    relevant = {k: settings[k] for k in SETTING_KEYS}
    relevant.update({k: settings[k] for k in OPTION_KEYS if settings.get(k)})
    relevant.update(model_dependencies(model, settings, choices))
    text = json.dumps({"cohort": cohort, "usable": rule, "model": model, "arm": arm, "cache_version": CACHE_VERSION,
                       **relevant}, sort_keys=True, default=str)
    return hashlib.sha1(text.encode()).hexdigest()[:16]


def device_for(resources) -> str:
    return "cuda" if resources.device == "cuda" and resources.n_gpus else "cpu"  # one GPU per worker process (D73)


def cohort_jobs(key: str, dataset: str, rule: str, order: int, anchors: pd.DataFrame, arms_of: dict, settings: dict,
                choices: dict, runs: Path, recompute, derived: Path, device: str, stream0: int) -> list[dict]:
    """One job per patient, anchor and model of one cohort (``arms_of``: model -> arms); every arm and the
    patient's persistence cached under ``runs/cache/<dataset>__<cohort>/``."""
    folder = key.replace("/", "__")
    jobs = []
    for r in anchors.itertuples():
        name = f"{r.patient_id}__{r.period}__{int(r.anchor_day)}"
        reference = {"path": str(runs / "cache" / folder / "persistence" / f"{name}.pkl"),
                     "key": cache_key(key, rule, "persistence", "reference", settings, choices), "recompute": recompute}
        for m, arms in arms_of.items():
            cache = {a: {"path": str(runs / "cache" / folder / m / a / f"{name}.pkl"),
                         "key": cache_key(key, rule, m, a, settings, choices), "recompute": recompute} for a in arms}
            jobs.append({"name": f"{key}__{name}__{m}", "dataset": dataset, "cohort": key, "order": order,
                         "usable": rule, "patient_id": r.patient_id, "period": r.period,
                         "anchor_day": int(r.anchor_day), "anchor": r.anchor, "derived_root": derived,
                         "settings": settings, "model": m, "arms": list(arms), "interval": None, "cache": cache,
                         "reference": reference, "stream": stream0 + len(jobs), "device": device})
    return jobs


def job_order(job: dict) -> tuple:
    """Cohorts in config order; within one, the networks first (the longest jobs), LightGBM and the rest last."""
    return job["order"], split_name(job["model"])[0] not in FAMILIES or job["model"].startswith("lightgbm")


def run_job(job: dict) -> dict:
    """The arms of one model, cohort and patient that have no valid cached result; each is cached."""
    from .mechanisms import evaluate_mechanisms

    started = time.time()
    todo = [a for a in job["arms"] if not cache_valid(job["cache"][a])]
    reference = not cache_valid(job["reference"])
    if not todo and not reference:
        return {"job": job["name"], "status": "cached", "arms": [], "seconds": 0.0, "warnings": "", "error": ""}
    with WarningRecorder() as recorder:
        result = evaluate_mechanisms({**job, "arms": todo, "with_reference": reference})
    status = {"job": job["name"], "status": "error" if result.error else "ok", "arms": todo,
              "seconds": round(time.time() - started, 1), "warnings": recorder.text(), "error": result.error}
    if result.error:
        return status
    not_cached = []
    for arm in todo:
        days = result.days[result.days["policy"] == arm] if not result.days.empty else result.days
        clarke = result.clarke.subset(lambda key, arm=arm: key["policy"] == arm)
        info = [row for row in result.info if row.get("policy") == arm]
        if any(retryable(row) for row in info):  # an unexpected failure (e.g. CUDA): computed again next run
            not_cached.append(arm)
            continue
        refits = {k: v for k, v in result.refits.items() if k.split("|")[0] == arm}
        write_cache(job["cache"][arm], {"days": days, "clarke": clarke, "info": info, "refits": refits})
    if reference:
        days = result.days[result.days["policy"] == "reference"] if not result.days.empty else result.days
        write_cache(job["reference"], {"days": days, "info": [], "refits": {},
                                       "clarke": result.clarke.subset(lambda key: key["policy"] == "reference")})
    status["not_cached"] = not_cached
    status["retry_errors"] = sorted({row["error"] for row in result.info if retryable(row)})
    status["cuda_fallbacks"] = sum(1 for row in result.info if row.get("cuda_error"))
    return status


def task_table(statuses: list[dict]) -> pd.DataFrame:
    """One row per job: status, arms run, failures, seconds and warnings."""
    return pd.DataFrame([{"job": r["job"], "status": r["status"], "arms_run": " ".join(r["arms"]),
                          "not_cached": " ".join(r.get("not_cached", [])),
                          "retry_errors": " | ".join(r.get("retry_errors", [])),
                          "cuda_fallbacks": r.get("cuda_fallbacks", 0), "seconds": r["seconds"],
                          "warnings": r["warnings"],
                          "error": r["error"].strip().splitlines()[-1] if r["error"] else "", "traceback": r["error"]}
                         for r in statuses])


def collect(jobs: list[dict]) -> tuple[pd.DataFrame, list, ClarkeAccumulator, list]:
    """Cached results of every job, with the cohort on every row and Clarke panel."""
    frames, infos, missing = [], [], []
    clarke = ClarkeAccumulator(["cohort", "model", "policy", "horizon_min", "period"], sample_size=20000)
    seen = set()
    for job in jobs:
        entries = [(a, job["cache"][a]) for a in job["arms"]]
        if job["reference"]["path"] not in seen:
            entries.append(("reference", job["reference"]))
            seen.add(job["reference"]["path"])
        for arm, entry in entries:
            cached = read_cache(entry)
            if cached is None:
                missing.append(f"{job['name']}:{arm}")
                continue
            if not cached["days"].empty:
                frames.append(cached["days"].assign(cohort=job["cohort"]))
            infos += [{**row, "cohort": job["cohort"]} for row in cached["info"]]
            clarke.merge(cached["clarke"].tagged(cohort=job["cohort"]))
    days = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return days, infos, clarke, missing


def write_fit_log(fit_log: pd.DataFrame, path: Path, keys: list[str]) -> None:
    """The fit log as parquet (``keys`` first; dates as dates, text as strings)."""
    columns = [c for c in list(keys) + list(FIT_LOG_COLUMNS) if c in fit_log]
    table = fit_log[columns].copy()
    for c in ("fit_day", "train_start"):
        if c in table:
            table[c] = pd.to_datetime(table[c].astype(str), errors="coerce")
    for c in table.columns:
        if table[c].dtype == object:
            table[c] = table[c].astype("string")
    table.to_parquet(path, index=False)
