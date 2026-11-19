"""20 - Stage 1: how personal models age (frozen, fresh rolling and fresh growing versions).

Runs on the screen group of the core dataset, from every available anchor (14,
42, 70 and 98 days after day 0, D51) of the core period (D52), and tests every
model on the 84 days after the anchor in three versions (HANDOFF 7.3, D65):

- frozen: fitted once on the 14 days before the anchor;
- rolling: refitted every day (every week for networks) on the last 14 days;
- growing: refitted as often on every day since the start of the frozen window.

The models are AR(3), ARX and the learned models with the starting point
chosen in stage 0 (D63: fine-tuned from the population models of script 11),
each with its glucose-only twin (D90: same versions and refits, reading no
insulin, carbohydrates or time of day; the figures show the full models). Every
forecast is compared with persistence on the same origins, so the daily skill
does not mix ageing with how hard the days are. A refit at the start of a day
learns only from targets before that day (HANDOFF 7.1).

This script writes the runs and descriptive tables: weekly skill curves,
the effects of recency (rolling - frozen) and amount of data (growing -
rolling), a per-patient slope of the frozen skill over the weeks, and the cost
of every version. Script 21 fits the mixed models (useful life, decision
point of stage 1). Reports RMSE and Clarke zones for every model and version
(rule R6). One job per patient and anchor on all cores but two, networks on
the GPUs in turn (rule R5). The confirm group is never touched.

Derived data:
    <derived_data>/runs/20_stage1_ageing_runs/day_metrics.parquet   per patient, anchor, model, version, horizon, day

Outputs (results/20_stage1_ageing_runs/):
    ageing_by_week.csv       per period, model, version, horizon and week since the anchor: mean patient skill, CI
    differences_by_week.csv  per period, model, horizon and week: fresh version - frozen, paired by patient, CI
    version_effects.csv      per period, model and horizon: skill of each version and the recency and data effects
    frozen_slopes.csv        per period, model and horizon: change in frozen skill per week (patient slopes), CI
    cost.csv                 per model and version: fits, seconds per fit, total seconds, training samples
    metrics_overall.csv      per period, model, version and horizon: n, RMSE, MAE, MARD, bias, Clarke zones (R6)
    clarke_zones.csv / clarke_histogram.csv / clarke_points_sample.csv   (R6)
    jobs.csv                 the patient-anchor jobs
    fit_log.csv              every fit: window, samples, seconds, rounds or steps
    task_log.csv
    ageing_curves.png
    ageing_differences.png
    version_effects.png
    clarke_grids.png
    summary.md
    run_manifest.json

Usage:
    python scripts/20_stage1_ageing_runs.py
    python scripts/20_stage1_ageing_runs.py --periods post_randomization whole     # add the sensitivity period
    python scripts/20_stage1_ageing_runs.py --models ar3 lightgbm_ft --patients 5   # a quick partial run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from t1dretrain.envguard import ensure_project_env  # noqa: E402  (standard library only)

ensure_project_env(__file__)  # re-runs this script inside the project environment if needed

from t1dretrain.codecheck import verify_code  # noqa: E402  (standard library only)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from t1dretrain.compute import detect_resources, parallel_map, share_gpus  # noqa: E402
from t1dretrain.config import CONFIG_DIR, derived_data_dir, load_project_config  # noqa: E402
from t1dretrain.data.pipeline import WarningRecorder  # noqa: E402
from t1dretrain.data.splits import group_patients  # noqa: E402
from t1dretrain.data.store import ProcessedStore  # noqa: E402
from t1dretrain.eval.metrics import ClarkeAccumulator, plot_clarke  # noqa: E402
from t1dretrain.eval.stage0_report import (DATA_ERRORS, FIT_NOTE, PERIOD_LABELS, DMDTable, finish,  # noqa: E402
                                           fit_outcomes, label, read_noise_floor)
from t1dretrain.eval.stats import bootstrap_mean  # noqa: E402
from t1dretrain.outputs import ScriptOutput, markdown_table  # noqa: E402
from t1dretrain.plotting import TEXT_SECONDARY, set_style  # noqa: E402
from t1dretrain.sim.stage0 import (FAMILIES, STATISTICAL_MODELS, model_dependencies, per_model,  # noqa: E402
                                   split_name, split_variant, twin_available, twin_families, with_twins)

SCRIPT_ID = "20_stage1_ageing_runs"
SUMS = ["n", "sse", "sae", "sape", "se", "zone_A", "zone_B", "zone_C", "zone_D", "zone_E"]
VERSION_COLORS = {"frozen": "#d1495b", "rolling": "#2a78d6", "growing": "#1baf7a"}
EFFECTS = {"recency (rolling - frozen)": ("rolling", "frozen"), "data (growing - rolling)": ("growing", "rolling"),
           "both (growing - frozen)": ("growing", "frozen")}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--paths", type=Path, default=CONFIG_DIR / "paths.yaml", help="paths file (config/paths.yaml)")
    parser.add_argument("--periods", nargs="*", default=None, help="periods (default: stage1.periods)")
    parser.add_argument("--models", nargs="*", default=None, help="subset of stage1.models")
    parser.add_argument("--patients", type=int, default=None, help="only the first N patients (quick runs)")
    parser.add_argument("--recompute", action="store_true", help="ignore the cached results of earlier runs")
    parser.add_argument("--skip-code-check", action="store_true",
                        help="run even if the code differs from the released manifest")
    return parser.parse_args()


def run_job(job: dict) -> dict:
    """Run the models of one patient-anchor that have no valid cached result, and cache each one (with
    persistence) under <derived>/runs/20_stage1_ageing_runs/cache/<model>/."""
    from t1dretrain.sim.ageing import evaluate_anchor

    started = __import__("time").time()
    todo = [m for m in job["models"] if not cache_valid(job["cache"][m])]
    if not todo:
        return {"job": job["name"], "status": "cached", "models": [], "seconds": 0.0, "warnings": "", "error": ""}
    with WarningRecorder() as recorder:
        result = evaluate_anchor({**job, "settings": {**job["settings"], "models": todo}})
    status = {"job": job["name"], "status": "error" if result.error else "ok", "models": todo,
              "seconds": round(__import__("time").time() - started, 1), "warnings": recorder.text(),
              "error": result.error}
    if result.error:
        return status
    not_cached = []
    for model in todo + ["persistence"]:
        days = result.days[result.days["model"] == model] if not result.days.empty else result.days
        clarke = result.clarke.subset(lambda key, model=model: key["model"] == model)
        info = [row for row in result.info if row.get("model") == model]
        if any(retryable(row) for row in info):  # an unexpected failure (e.g. CUDA): computed again next run
            not_cached.append(model)
            continue
        write_cache(job["cache"][model], {"days": days, "clarke": clarke, "info": info})
    status["not_cached"] = not_cached
    status["cuda_fallbacks"] = sum(1 for row in result.info if row.get("cuda_error"))
    return status




def retryable(row: dict) -> bool:
    """A failed fit whose cause is not the data (a window without usable samples fails the same way every time)."""
    error = row.get("error")
    return isinstance(error, str) and bool(error) and not error.startswith(DATA_ERRORS)


def cache_key(model: str, settings: dict, choices: dict) -> str:
    """What a cached result depends on: the model, its starting point (choices of its family), the stage-1
    settings and the version of the simulator. A new population model or a change of settings recomputes it."""
    import hashlib
    import json

    from t1dretrain.sim.ageing import CACHE_VERSION

    relevant = {k: settings[k] for k in ("horizons_min", "lookback_hours", "origin_stride_min", "train_stride_min",
                                         "validation_days", "min_window_share", "recent_complete_min", "window_days",
                                         "test_days", "versions", "seeds")}
    relevant["refresh_days"] = per_model(settings["refresh_days"], model, 1)
    relevant.update(model_dependencies(model, settings, choices))
    text = json.dumps({"model": model, "cache_version": CACHE_VERSION, **relevant}, sort_keys=True, default=str)
    return hashlib.sha1(text.encode()).hexdigest()[:16]


def cache_valid(entry: dict) -> bool:
    import pickle

    path = Path(entry["path"])
    if not path.exists():
        return False
    try:
        with open(path, "rb") as handle:
            content = pickle.load(handle)
    except Exception:  # noqa: BLE001
        return False
    if entry.get("recompute") and content.get("run") != entry["recompute"]:  # --recompute: only this run's results
        return False
    # results of an earlier release may hold fits that failed for a reason other than the data: recompute them
    return content.get("key") == entry["key"] and not any(retryable(row) for row in content.get("info", []))


def write_cache(entry: dict, content: dict) -> None:
    import pickle

    path = Path(entry["path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.with_suffix(".tmp"), "wb") as handle:
        pickle.dump({"key": entry["key"], "run": entry.get("recompute"), **content}, handle)
    path.with_suffix(".tmp").replace(path)


def read_cache(entry: dict) -> dict | None:
    import pickle

    if not cache_valid(entry):
        return None
    with open(entry["path"], "rb") as handle:
        return pickle.load(handle)


def device_for(k: int, resources) -> str:
    return "cuda" if resources.device == "cuda" and resources.n_gpus else "cpu"  # one GPU per worker process (D73)


def main() -> None:
    import time

    run_start = time.time()  # --recompute: marks the results of this run (sim.cache)
    args = parse_args()
    release = verify_code(strict=not args.skip_code_check)
    config = load_project_config()
    resources = detect_resources(config.get("compute", {}).get("reserve_cores"))
    print(resources.describe())
    print(f"GPU sharing: at most {share_gpus(config)} processes compute on one GPU at a time (D73)")
    out = ScriptOutput(SCRIPT_ID, resources=resources)
    set_style()
    derived = derived_data_dir(args.paths)
    core = config.get("core_dataset", "replace_bg")
    stage, stage1 = config["stage0"], config["stage1"]
    population_dir = derived / "models" / core / "population"
    candidates = [m for m in with_twins(stage1["models"], twin_families(config))  # each with its twin (D90)
                  if not args.models or m in args.models or split_variant(m)[0] in args.models]
    models = [m for m in candidates if twin_available(m, population_dir, stage1["seeds"])]
    if len(models) < len(candidates):
        print(f"Left out (no glucose-only population model of script 11): {sorted(set(candidates) - set(models))}")
    needs_population = [m for m in models if split_name(m)[0] in FAMILIES and split_name(m)[1] != "scratch"]
    choices = {}
    if (population_dir / "choices.yaml").exists():
        choices = yaml.safe_load((population_dir / "choices.yaml").read_text())
    elif needs_population:
        raise SystemExit(f"{population_dir / 'choices.yaml'} not found: run scripts/11_pretrain_population.py first")
    periods = args.periods or list(stage1["periods"])
    settings = {"horizons_min": [int(h) for h in config["horizons_min"]], "lookback_hours": float(config["lookback_hours"]),
                "origin_stride_min": float(config["origin_stride_min_screen"]),
                "train_stride_min": float(stage["train_stride_min"]), "validation_days": int(stage["validation_days"]),
                "min_window_share": float(stage["min_window_share"]),
                "recent_complete_min": float(stage["recent_complete_min"]),
                "window_days": int(config["anchors"]["train_window_days"]),
                "test_days": int(config["anchors"]["max_age_days"]),
                "clarke_point_sample": int(config["evaluation"]["clarke_point_sample"]),
                "models": models, "versions": list(stage1["versions"]), "seeds": [int(s) for s in stage1["seeds"]],
                "refresh_days": {k: int(v) for k, v in stage1["refresh_days"].items()},
                "population_dir": str(population_dir),
                "finetune": {f: (choices.get(f) or {}).get("finetune") for f in FAMILIES}}
    for f in FAMILIES:
        settings[f] = (choices.get(f) or {}).get("scratch") or stage.get(f)
    for m in STATISTICAL_MODELS:
        settings[m] = stage.get(m) or {}

    patients = group_patients(core, stage1.get("group", "screen"))
    if args.patients:
        patients = patients[: args.patients]
    anchors = ProcessedStore(core, root=derived, stage="clean").read_extra("anchors")
    anchors = anchors[(anchors["cohort"] == "core") & anchors["patient_id"].isin(patients) &
                      anchors["period"].isin(periods) & anchors["available"].astype(bool)].copy()
    anchors["anchor"] = pd.to_datetime(anchors["day0"]) + pd.to_timedelta(anchors["anchor_day"], unit="D")
    anchors = anchors.sort_values(["patient_id", "period", "anchor_day"]).reset_index(drop=True)
    runs = derived / "runs" / SCRIPT_ID
    keys = {m: cache_key(m, settings, choices) for m in models + ["persistence"]}
    jobs = []
    for k, r in enumerate(anchors.itertuples()):
        name = f"{r.patient_id}__{r.period}__{int(r.anchor_day)}"
        cache = {m: {"path": str(runs / "cache" / m / f"{name}.pkl"), "key": keys[m],
                     "recompute": run_start if args.recompute else None}
                 for m in models + ["persistence"]}
        jobs.append({"name": name, "dataset": core, "patient_id": r.patient_id, "period": r.period,
                     "anchor_day": int(r.anchor_day), "anchor": r.anchor, "derived_root": derived, "settings": settings,
                     "stream": k, "device": device_for(k, resources), "models": models, "cache": cache})
    print(f"{len(jobs)} patient-anchor jobs ({len(patients)} patients, periods {periods}, models {models}); "
          f"results cached in {runs / 'cache'}")
    statuses = parallel_map(run_job, jobs, resources.n_workers, label="patient-anchors")

    task_log = pd.DataFrame([{"job": r["job"], "status": r["status"], "models_run": " ".join(r["models"]),
                              "not_cached": " ".join(r.get("not_cached", [])),
                              "cuda_fallbacks": r.get("cuda_fallbacks", 0), "seconds": r["seconds"],
                              "warnings": r["warnings"],
                              "error": r["error"].strip().splitlines()[-1] if r["error"] else "",
                              "traceback": r["error"]} for r in statuses])
    frames, infos = [], []
    clarke = ClarkeAccumulator(["model", "version", "horizon_min", "period"], sample_size=settings["clarke_point_sample"])
    missing = []
    for job in jobs:
        for m in models + ["persistence"]:
            cached = read_cache(job["cache"][m])
            if cached is None:
                missing.append(f"{job['name']}:{m}")
                continue
            frames.append(cached["days"])
            infos += cached["info"]
            clarke.merge(cached["clarke"])
    days = pd.concat([f for f in frames if not f.empty], ignore_index=True) if frames else pd.DataFrame()
    runs.mkdir(parents=True, exist_ok=True)
    days.to_parquet(runs / "day_metrics.parquet", index=False)
    fit_log = pd.DataFrame(infos)
    replicates, seed = int(stage["bootstrap_replicates"]), int(config["random_seed"])
    dmd = read_dmd(Path(out.dir).parent / "12_stage0_population_transfer" / "noise_floor.csv",
                   [m for m in models if split_name(m)[0] in FAMILIES])

    weekly = ageing_by_week(days, replicates, seed) if not days.empty else pd.DataFrame()
    gaps = differences_by_week(days, replicates, seed) if not days.empty else pd.DataFrame()
    effects = version_effects(days, dmd, replicates, seed) if not days.empty else pd.DataFrame()
    slopes = frozen_slopes(days, replicates, seed) if not days.empty else pd.DataFrame()
    cost = cost_table(fit_log)
    overall = finish(days.groupby(["period", "model", "version", "horizon_min"], as_index=False)[SUMS].sum()).round(4) \
        if not days.empty else pd.DataFrame()
    histogram = clarke.histogram()

    out.csv(weekly, "ageing_by_week")
    out.csv(gaps, "differences_by_week")
    out.csv(effects, "version_effects")
    out.csv(slopes, "frozen_slopes")
    out.csv(cost, "cost")
    out.csv(overall, "metrics_overall")
    out.csv(clarke.zones(), "clarke_zones")
    out.csv(histogram, "clarke_histogram")
    out.csv(clarke.sample(), "clarke_points_sample")
    out.csv(anchors.drop(columns=["dataset"], errors="ignore"), "jobs")
    out.csv(fit_log, "fit_log")
    out.csv(fit_outcomes(fit_log), "fit_outcomes")
    out.csv(task_log, "task_log")
    shown_models = [m for m in models if not weekly.empty and m in set(weekly["model"]) and not split_variant(m)[1]]
    # the figures show the full models; their glucose-only twins (D90) are in the tables and in script 21
    if shown_models:
        period = periods[0]
        out.figure(plot_ageing(weekly, shown_models, period, dmd), "ageing_curves")
        if not gaps.empty:
            out.figure(plot_gaps(gaps, shown_models, period, dmd), "ageing_differences")
        if not effects.empty:
            out.figure(plot_effects(effects[effects["model"].isin(shown_models)], period), "version_effects")
        out.figure(plot_grids(histogram, shown_models, settings["horizons_min"], period), "clarke_grids")
    out.text(write_summary(weekly, effects, slopes, cost, overall, fit_log, task_log, missing, settings, periods,
                           len(patients), len(jobs), dmd, choices, release, resources), "summary.md")
    out.manifest(settings={**{k: v for k, v in settings.items() if k != "population_dir"}, "periods": periods,
                           "patients": len(patients), "jobs": len(jobs), "population_models": str(population_dir),
                           "choices_release": choices.get("release")},
                 notes="One job per patient and anchor; day metrics in <derived>/runs/20_stage1_ageing_runs.")
    print(f"Done. Results in {out.dir}")


# ============================================================================ tables


def read_dmd(path: Path, learned: list[str]) -> DMDTable:
    """DMD of each model of stage 1 (D75): a learned model's own DMD in script 12; AR(3) and ARX (no seed noise)
    take the smallest DMD of the stage-1 full learned models (twins do not set it, D90). Empty (no ties) if script
    12 has not run here."""
    return DMDTable(read_noise_floor(path), learned)


def with_skill(days: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """RMSE of every model and version over ``keys`` (sums pooled), and skill against persistence on the same
    patient, anchor, days and horizon."""
    grouped = days.groupby(keys + ["model", "version", "seed"], as_index=False)[["n", "sse"]].sum()
    grouped["rmse"] = np.sqrt(grouped["sse"] / grouped["n"])
    reference = grouped[grouped["model"] == "persistence"].set_index(keys)["rmse"]
    grouped = grouped[grouped["model"] != "persistence"].copy()
    grouped["skill"] = 1 - grouped["rmse"] / grouped.set_index(keys).index.map(reference)
    return grouped


def ageing_by_week(days: pd.DataFrame, replicates: int, seed: int) -> pd.DataFrame:
    """Weekly skill per patient (forecasts of the week pooled over the patient's anchors), then the mean over
    patients with a bootstrap interval."""
    d = days.copy()
    d["week"] = (d["days_since_anchor"] // 7).astype(int)
    skill = with_skill(d, ["period", "patient_id", "anchor_day", "horizon_min", "week"])
    per_patient = skill.groupby(["period", "model", "version", "horizon_min", "week", "patient_id"])["skill"].mean()
    rows = []
    for (period, model, version, horizon, week), g in per_patient.groupby(level=[0, 1, 2, 3, 4]):
        values = g.dropna().to_numpy()
        if len(values) < 3:
            continue
        mean, low, high = bootstrap_mean(values, replicates, seed)
        rows.append({"period": period, "model": model, "version": version, "horizon_min": horizon, "week": week,
                     "patients": len(values), "skill_mean": mean, "skill_low": low, "skill_high": high})
    return pd.DataFrame(rows).round(4)


def differences_by_week(days: pd.DataFrame, replicates: int, seed: int) -> pd.DataFrame:
    """Per week since the anchor, the paired difference in weekly skill (fresh version - frozen) per patient, with a
    bootstrap interval: the shared ups and downs of hard weeks cancel out, so the gap of the frozen model shows."""
    d = days.copy()
    d["week"] = (d["days_since_anchor"] // 7).astype(int)
    skill = with_skill(d, ["period", "patient_id", "anchor_day", "horizon_min", "week"])
    wide = skill.groupby(["period", "model", "horizon_min", "week", "patient_id", "version"])["skill"].mean() \
        .unstack("version")
    rows = []
    for (period, model, horizon, week), g in wide.groupby(level=[0, 1, 2, 3]):
        for version in ("rolling", "growing"):
            if version not in g or "frozen" not in g:
                continue
            diff = (g[version] - g["frozen"]).dropna().to_numpy()
            if len(diff) < 3:
                continue
            mean, low, high = bootstrap_mean(diff, replicates, seed)
            rows.append({"period": period, "model": model, "horizon_min": horizon, "week": week, "version": version,
                         "patients": len(diff), "difference": mean, "low": low, "high": high})
    return pd.DataFrame(rows).round(4)


def version_effects(days: pd.DataFrame, dmd: DMDTable, replicates: int, seed: int) -> pd.DataFrame:
    """Mean skill of each version over the whole test span, and the differences between versions per patient
    (recency: rolling - frozen; data: growing - rolling), with intervals and the tie rule (D61) with the DMD of
    the model (D75)."""
    skill = with_skill(days, ["period", "patient_id", "anchor_day", "horizon_min"])
    per_patient = skill.groupby(["period", "model", "horizon_min", "patient_id", "version"])["skill"].mean() \
        .unstack("version")
    rows = []
    for (period, model, horizon), g in per_patient.groupby(level=[0, 1, 2]):
        row = {"period": period, "model": model, "horizon_min": horizon, "patients": int(len(g))}
        for version in g.columns:
            row[f"skill_{version}"] = float(g[version].mean())
        floor = dmd.model(period, horizon, model)  # versions of one model: its own DMD (D75)
        for name, (a, b) in EFFECTS.items():
            if a not in g or b not in g:
                continue
            diff = (g[a] - g[b]).dropna().to_numpy()
            if len(diff) < 3:
                continue
            mean, low, high = bootstrap_mean(diff, replicates, seed)
            verdict = "tie" if np.isfinite(floor) and abs(mean) < floor else \
                "positive" if low > 0 else "negative" if high < 0 else "uncertain"
            rows.append({**row, "effect": name, "difference": mean, "low": low, "high": high,
                         "patients_positive": int((diff > 0).sum()), "dmd": floor, "verdict": verdict})
    return pd.DataFrame(rows).round(4)


def frozen_slopes(days: pd.DataFrame, replicates: int, seed: int) -> pd.DataFrame:
    """Per patient, the least-squares slope of the weekly frozen skill over the weeks since the anchor (skill
    lost per week when the model is not refitted); mean over patients with a bootstrap interval."""
    d = days[days["version"].isin(["frozen", "reference"])].copy()
    d["week"] = (d["days_since_anchor"] // 7).astype(int)
    skill = with_skill(d, ["period", "patient_id", "anchor_day", "horizon_min", "week"])
    weekly = skill.groupby(["period", "model", "horizon_min", "patient_id", "week"])["skill"].mean().reset_index()
    rows = []
    for (period, model, horizon), g in weekly.groupby(["period", "model", "horizon_min"]):
        slopes = []
        for _, p in g.groupby("patient_id"):
            p = p.dropna()
            if p["week"].nunique() >= 4:
                slopes.append(np.polyfit(p["week"], p["skill"], 1)[0])
        if len(slopes) < 3:
            continue
        mean, low, high = bootstrap_mean(np.array(slopes), replicates, seed)
        rows.append({"period": period, "model": model, "horizon_min": horizon, "patients": len(slopes),
                     "slope_per_week": mean, "low": low, "high": high,
                     "patients_declining": int((np.array(slopes) < 0).sum())})
    return pd.DataFrame(rows).round(5)


def cost_table(fit_log: pd.DataFrame) -> pd.DataFrame:
    if fit_log.empty or "fit_seconds" not in fit_log:
        return pd.DataFrame()
    ok = fit_log[fit_log["fit_seconds"].notna()]
    table = ok.groupby(["model", "version"]).agg(fits=("fit_seconds", "size"),
                                                  seconds_median=("fit_seconds", "median"),
                                                  seconds_total=("fit_seconds", "sum"),
                                                  train_samples_median=("train_samples", "median"),
                                                  train_days_median=("train_days", "median")).reset_index()
    jobs = ok.groupby(["model", "version"])[["patient_id", "anchor_day"]].apply(
        lambda g: len(g.drop_duplicates())).rename("patient_anchors").reset_index()
    table = table.merge(jobs, on=["model", "version"])
    table["fits_per_patient_anchor"] = table["fits"] / table["patient_anchors"]
    return table.round(3)


# ============================================================================ figures


def plot_ageing(weekly: pd.DataFrame, models: list[str], period: str, dmd: DMDTable) -> plt.Figure:
    w = weekly[weekly["period"] == period]
    horizons = sorted(w["horizon_min"].unique())
    fig, axes = plt.subplots(len(models), len(horizons), figsize=(4.8 * len(horizons), 3.4 * len(models)),
                             squeeze=False, sharex=True)
    for i, model in enumerate(models):
        for j, horizon in enumerate(horizons):
            ax = axes[i, j]
            for version, colour in VERSION_COLORS.items():
                v = w[(w["model"] == model) & (w["horizon_min"] == horizon) & (w["version"] == version)]
                if v.empty:
                    continue
                ax.plot(v["week"], v["skill_mean"], color=colour, lw=2, label=version)
                ax.fill_between(v["week"], v["skill_low"], v["skill_high"], color=colour, alpha=0.15, linewidth=0)
            floor = dmd.model(period, horizon, model)
            ax.set_title(f"{label(model)}, {horizon} min" + (f" (DMD {floor:.3f})" if np.isfinite(floor) else ""),
                         loc="left", fontsize=9)
            ax.axhline(0, color=TEXT_SECONDARY, lw=0.6)
            if i == len(models) - 1:
                ax.set_xlabel("weeks since the anchor")
        axes[i, 0].set_ylabel("skill over persistence")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"Stage 1: skill by week since the anchor, {PERIOD_LABELS.get(period, period)} (mean over patients, "
                 "95 % CI)", fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    return fig


def plot_gaps(gaps: pd.DataFrame, models: list[str], period: str, dmd: DMDTable) -> plt.Figure:
    g = gaps[gaps["period"] == period]
    horizons = sorted(g["horizon_min"].unique())
    fig, axes = plt.subplots(len(models), len(horizons), figsize=(4.8 * len(horizons), 3.0 * len(models)),
                             squeeze=False, sharex=True)
    for i, model in enumerate(models):
        for j, horizon in enumerate(horizons):
            ax = axes[i, j]
            floor = dmd.model(period, horizon, model)
            if np.isfinite(floor):
                ax.axhspan(-floor, floor, color=TEXT_SECONDARY, alpha=0.15, lw=0, label="DMD (tie)")
            for version in ("rolling", "growing"):
                v = g[(g["model"] == model) & (g["horizon_min"] == horizon) & (g["version"] == version)]
                if v.empty:
                    continue
                ax.plot(v["week"], v["difference"], color=VERSION_COLORS[version], lw=2, label=f"{version} - frozen")
                ax.fill_between(v["week"], v["low"], v["high"], color=VERSION_COLORS[version], alpha=0.15, linewidth=0)
            ax.axhline(0, color=TEXT_SECONDARY, lw=0.8)
            ax.set_title(f"{label(model)}, {horizon} min", loc="left", fontsize=9)
            if i == len(models) - 1:
                ax.set_xlabel("weeks since the anchor")
        axes[i, 0].set_ylabel("skill difference")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(f"Stage 1: what refitting adds each week, {PERIOD_LABELS.get(period, period)} (paired by patient, "
                 "95 % CI)", fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    return fig


def plot_effects(effects: pd.DataFrame, period: str) -> plt.Figure:
    e = effects[effects["period"] == period].copy()
    e["row"] = e["model"].map(label) + ", " + e["effect"]
    horizons = sorted(e["horizon_min"].unique())
    rows = list(dict.fromkeys(e["row"]))
    fig, axes = plt.subplots(1, len(horizons), figsize=(5.2 * len(horizons), 0.42 * len(rows) + 1.8), sharey=True,
                             squeeze=False)
    for ax, horizon in zip(axes[0], horizons):
        h = e[e["horizon_min"] == horizon].set_index("row").reindex(rows)
        y = np.arange(len(rows))[::-1]
        for yy, (_, r) in zip(y, h.iterrows()):
            if pd.isna(r["difference"]):
                continue
            colour = VERSION_COLORS[EFFECTS[r["effect"]][0]]
            ax.plot([r["low"], r["high"]], [yy, yy], color=colour, lw=2)
            ax.scatter(r["difference"], yy, color=colour, zorder=3, s=28,
                       marker="o" if r["verdict"] in ("positive", "negative") else "s")
        labelled = False
        for yy, (_, r) in zip(y, h.iterrows()):  # the tie band of each model (D75)
            if pd.notna(r["dmd"]):
                ax.fill_between([-r["dmd"], r["dmd"]], yy - 0.35, yy + 0.35, color=TEXT_SECONDARY, alpha=0.15, lw=0,
                                zorder=0, label=None if labelled else "DMD of the model (tie)")
                labelled = True
        if labelled:
            ax.legend(fontsize=8, loc="lower right")
        ax.axvline(0, color=TEXT_SECONDARY, lw=0.8)
        ax.set_yticks(y, rows, fontsize=8)
        ax.set_title(f"{horizon} min", loc="left", fontsize=10)
        ax.set_xlabel("difference in mean skill (95 % CI)")
    fig.suptitle(f"Effects of refitting, {PERIOD_LABELS.get(period, period)} (circle: beyond the DMD; square: tie or "
                 "uncertain)", fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    return fig


def plot_grids(histogram: pd.DataFrame, models: list[str], horizons: list[int], period: str) -> plt.Figure:
    panels = [(m, v) for m in models for v in ("frozen", "rolling")]
    fig, axes = plt.subplots(len(panels), len(horizons), figsize=(4.3 * len(horizons), 4.4 * len(panels)),
                             squeeze=False)
    for i, (model, version) in enumerate(panels):
        for j, horizon in enumerate(horizons):
            h = histogram[(histogram["model"] == model) & (histogram["version"] == version) &
                          (histogram["horizon_min"] == horizon) & (histogram["period"] == period)]
            plot_clarke(axes[i, j], histogram=h, title=f"{label(model)} {version}, {horizon} min")
            if j:
                axes[i, j].set_ylabel("")
            if i < len(panels) - 1:
                axes[i, j].set_xlabel("")
    fig.suptitle(f"Clarke Error Grid, stage 1, {PERIOD_LABELS.get(period, period)} (density, log scale)",
                 fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    return fig


# ============================================================================ summary


def write_summary(weekly, effects, slopes, cost, overall, fit_log, task_log, missing, settings, periods, n_patients,
                  n_jobs, dmd, choices, release, resources) -> str:
    lines = ["# 20 - Stage 1: how personal models age", "", f"Code release {release}.",
             f"Machine: {resources.describe()}",
             f"Screen group: {n_patients} patients, {n_jobs} patient-anchor jobs; periods {periods}. Models "
             f"{settings['models']}, versions {settings['versions']}, seeds {settings['seeds']}; window "
             f"{settings['window_days']} days; {settings['test_days']} test days from each anchor; refits every "
             f"{settings['refresh_days']} days. Population models from script 11 (release "
             f"{choices.get('release', '-')}). DMD from script 12: {'found' if len(dmd) else 'not found (no ties)'}.", ""]
    problems = []
    for _, row in task_log[task_log["status"] == "error"].iterrows():
        problems.append(f"{row['job']} failed: {row['error']}")
    for _, row in task_log[task_log["warnings"].fillna("") != ""].head(10).iterrows():
        problems.append(f"{row['job']} raised warnings: {row['warnings'][:300]}")
    for item in missing[:10]:
        problems.append(f"no result for {item}")
    if not fit_log.empty and "error" in fit_log:
        failed = fit_log[fit_log["error"].notna()]
        for (model, error), g in failed.groupby(["model", "error"]):
            problems.append(f"{model}: {len(g)} fits failed ({', '.join(g['patient_id'].unique()[:5])}): {error}")
    if not fit_log.empty and "cuda_error" in fit_log:
        for _, row in fit_log[fit_log["cuda_error"].notna()].head(5).iterrows():
            problems.append(f"{row['patient_id']}: {row['model']} fell back to the CPU ({row['cuda_error']})")
    lines += ["## Warnings", ""] + ([f"- {p}" for p in problems] or ["- none"]) + [""]
    counts = task_log["status"].value_counts().to_dict() if not task_log.empty else {}
    lines += [f"Jobs: {counts.get('ok', 0)} computed, {counts.get('cached', 0)} taken from the cache of earlier runs, "
              f"{counts.get('error', 0)} failed.", ""]
    check = fit_outcomes(fit_log)
    if not check.empty:
        lines += ["## Fits (every refit of every model; a run is complete when `other_failure` is 0)", "",
                  markdown_table(check), "", FIT_NOTE + " Failed fits are not cached: running the script again "
                  "computes only those.", ""]
    if not effects.empty:
        show = list(effects.columns)
        lines += ["## Effects of refitting (per patient, whole test span; tie = below the DMD)", "",
                  markdown_table(effects[show]), ""]
    if not slopes.empty:
        lines += ["## Frozen versions: skill change per week since the anchor", "", markdown_table(slopes), ""]
    if not weekly.empty:
        first_last = weekly[weekly["week"].isin([0, weekly["week"].max()])]
        lines += ["## Skill in the first and last week", "", markdown_table(first_last), ""]
    if not cost.empty:
        lines += ["## Cost of each version", "", markdown_table(cost), ""]
    if not overall.empty:
        show = ["period", "model", "version", "horizon_min", "n", "rmse", "mae", "mard_pct", "bias", "clarke_A_pct",
                "clarke_B_pct", "clarke_AB_pct", "clarke_C_pct", "clarke_D_pct", "clarke_E_pct"]
        lines += ["## Pooled forecasts (R6)", "", markdown_table(overall[show]), ""]
    lines += ["## Checks to look at", "",
              "- `ageing_curves.png`: the frozen line should start with the others (same fit in week 0) and drift "
              "below them if models age; rolling and growing stay flat.",
              "- `ageing_differences.png`: the same paired by patient and week; a frozen model that ages shows a gap "
              "that grows with the weeks.",
              "- `version_effects.png`: recency (rolling - frozen) and data (growing - rolling) against the DMD.",
              "- `cost.csv`: refits per patient-anchor (84 daily, 12 weekly) and seconds per refit.", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
