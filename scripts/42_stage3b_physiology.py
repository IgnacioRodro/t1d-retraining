"""42 - Stage 3b: insulin action duration and carbohydrate absorption, for everyone and for each patient (D89).

The model inputs IOB, insulin activity, COB and carbohydrate absorption come
from one curve for every patient (D54: insulin acting over 360 min with its
peak at 75 min; carbohydrates absorbed over 180 min). This script computes them
again with every combination of ``stage3b`` (action duration x peak x
absorption time) on the screen group of the core dataset, from the anchor of
stage 3 (day 14 after randomisation, 84 test days, growing window), and asks
(HANDOFF 7.5; ``sim.physiology``):

- **for everyone**: how much the skill of ARX (updated at every sample without
  forgetting, D87) depends on the curves (``fixed_<set>`` arms, each against
  the default set);
- **for each patient**: whether the patient's own curves, chosen each day from
  its past forecasts only (``personal``: the set with the best skill so far),
  beat the default, how far that is from the best set on the test days
  themselves (the oracle, an upper bound only), and how often the choice
  changes;
- **with LightGBM** (restart from the population model at its optimal interval
  of script 31, the best mechanism of stage 3): ``restart_personal`` refits
  with ARX's choice of the day against ``restart`` with the default curves.

Every arm is scored on the same origins (RMSE and Clarke zones, rule R6), with
paired differences per patient and the tie rule of D61 with the model's DMD
(D75). Results are cached per patient, so a later run computes only what is new
or changed (``--recompute`` ignores the cache). One job per patient on all cores
but two (rule R5; LightGBM on the CPU). The confirm group is never touched.

Inputs:
    <derived_data>/features/<core>/grid/        the saved inputs (default curves; checked against the recomputed)
    <derived_data>/processed/<core>/insulin/ and carbs/   delivered insulin and logged carbohydrates (script 02)
    <derived_data>/runs/31_stage2_policy_analysis/confirmation.yaml   LightGBM's refit interval
    results/12_stage0_population_transfer/noise_floor.csv   (DMD, D75)

Derived data:
    <derived_data>/runs/42_stage3b_physiology/day_metrics.parquet and fit_log.parquet
    <derived_data>/runs/42_stage3b_physiology/cache/<patient-anchor>.pkl

Outputs (results/42_stage3b_physiology/):
    fixed_sets.csv           ARX: per set and horizon, mean skill and the paired difference with the default set
    arm_summary.csv          per model, arm and horizon: patients, skill and RMSE with intervals
    vs_default.csv           personal and the oracle against the default set (ARX); restart_personal against
                             restart and restart against never (LightGBM): paired, verdict with the DMD
    oracle.csv               per patient: the best set on the test days, the last set chosen from the past, gains
    choices.csv              per patient and test day: the set chosen from the past, its score and the default's
    choice_summary.csv       per set: the patients whose last choice it is and whose oracle it is
    metrics_overall.csv      per model, arm and horizon: n, RMSE, MAE, MARD, bias, Clarke zones (R6)
    clarke_zones.csv / clarke_histogram.csv / clarke_points_sample.csv   (R6; points of the default and personal arms)
    fit_log.csv / task_log.csv / jobs.csv
    fixed_sets.png           ARX: difference with the default set per action duration and absorption time
    choices.png              the sets chosen from the past (last day) and the oracle sets, per patient
    personal.png             per patient: gain of the personal curves over the default ones (ARX and LightGBM)
    summary.md
    run_manifest.json

Usage:
    python scripts/42_stage3b_physiology.py
    python scripts/42_stage3b_physiology.py --patients 4 --no-lightgbm   # a quick partial run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from t1dretrain.envguard import ensure_project_env  # noqa: E402  (standard library only)

ensure_project_env(__file__)  # re-runs this script inside the project environment if needed

from t1dretrain.codecheck import verify_code  # noqa: E402  (standard library only)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from t1dretrain.compute import detect_resources, parallel_map  # noqa: E402
from t1dretrain.config import CONFIG_DIR, derived_data_dir, load_project_config  # noqa: E402
from t1dretrain.data.pipeline import WarningRecorder  # noqa: E402
from t1dretrain.data.splits import group_patients  # noqa: E402
from t1dretrain.data.store import ProcessedStore  # noqa: E402
from t1dretrain.eval import policy_analysis as pa  # noqa: E402
from t1dretrain.eval import policy_report as pr  # noqa: E402
from t1dretrain.eval.metrics import ClarkeAccumulator  # noqa: E402
from t1dretrain.eval.stage0_report import (FIT_NOTE, PERIOD_LABELS, SUMS, DMDTable, finish, fit_outcomes,  # noqa: E402
                                           label, read_noise_floor)
from t1dretrain.outputs import ScriptOutput, markdown_table  # noqa: E402
from t1dretrain.plotting import TEXT_SECONDARY, set_style  # noqa: E402
from t1dretrain.sim.cache import cache_valid, read_cache, retryable, write_cache  # noqa: E402
from t1dretrain.sim.physiology import CACHE_VERSION, default_set, parameter_sets  # noqa: E402
from t1dretrain.sim.stage0 import FAMILIES, STATISTICAL_MODELS, model_dependencies, split_name  # noqa: E402

SCRIPT_ID = "42_stage3b_physiology"
FIT_KEYS = pr.FIT_KEYS
DIVERGING = ["#eb6834", "#f7f6f2", "#2a78d6"]  # worse than the default, as good, better


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--paths", type=Path, default=CONFIG_DIR / "paths.yaml", help="paths file (config/paths.yaml)")
    parser.add_argument("--patients", type=int, default=None, help="only the first N patients (quick runs)")
    parser.add_argument("--no-lightgbm", action="store_true", help="ARX only (no LightGBM arms)")
    parser.add_argument("--recompute", action="store_true", help="ignore the cached results of earlier runs")
    parser.add_argument("--skip-code-check", action="store_true",
                        help="run even if the code differs from the released manifest")
    return parser.parse_args()


# ============================================================================ jobs and cache


def run_job(job: dict) -> dict:
    """Every arm of one patient and anchor, unless a valid cached result exists; the result is cached."""
    import time

    from t1dretrain.sim.physiology import evaluate_physiology

    started = time.time()
    if cache_valid(job["cache"]):
        return {"job": job["name"], "status": "cached", "seconds": 0.0, "warnings": "", "error": ""}
    with WarningRecorder() as recorder:
        result = evaluate_physiology(job)
    status = {"job": job["name"], "status": "error" if result.error else "ok",
              "seconds": round(time.time() - started, 1), "warnings": recorder.text(), "error": result.error}
    if result.error:
        return status
    retry = sorted({row["error"] for row in result.info if retryable(row)})
    if not retry:
        write_cache(job["cache"], {"days": result.days, "clarke": result.clarke, "info": result.info,
                                   "refits": result.refits})
    status["retry_errors"] = retry
    return status


def cache_key(settings: dict, features: dict, sets: list, lightgbm: str | None, choices: dict,
              interval: int) -> str:
    """What a cached result depends on: the sets, the curves of script 05, the settings, LightGBM's starting
    point and interval, and the simulator."""
    import hashlib
    import json

    relevant = {k: settings[k] for k in ("horizons_min", "lookback_hours", "origin_stride_min", "train_stride_min",
                                         "validation_days", "min_window_share", "recent_complete_min", "window_days",
                                         "test_days", "seeds", "arx", "forgetting")}
    relevant["features"] = {k: features.get(k) for k in ("insulin_action_min", "insulin_peak_min",
                                                          "insulin_delay_min", "long_acting_spread_h",
                                                          "carb_absorption_min")}
    relevant["sets"] = [s.name for s in sets]
    if lightgbm:
        relevant["lightgbm"] = {"model": lightgbm, "interval": int(interval),
                                **model_dependencies(lightgbm, settings, choices)}
    text = json.dumps({"cache_version": CACHE_VERSION, **relevant}, sort_keys=True, default=str)
    return hashlib.sha1(text.encode()).hexdigest()[:16]


# ============================================================================ main


def main() -> None:
    import time

    run_start = time.time()  # --recompute: marks the results of this run (sim.cache)
    args = parse_args()
    release = verify_code(strict=not args.skip_code_check)
    config = load_project_config()
    resources = detect_resources(config.get("compute", {}).get("reserve_cores"))
    print(resources.describe())
    out = ScriptOutput(SCRIPT_ID, resources=resources)
    set_style()
    derived = derived_data_dir(args.paths)
    core = config.get("core_dataset", "replace_bg")
    stage, cfg, features = config["stage0"], config["stage3b"], config["features"]
    period = list(cfg["periods"])[0]
    population_dir = derived / "models" / core / "population"
    choices = yaml.safe_load((population_dir / "choices.yaml").read_text()) \
        if (population_dir / "choices.yaml").exists() else {}
    settings = {"horizons_min": [int(h) for h in config["horizons_min"]], "lookback_hours": float(config["lookback_hours"]),
                "origin_stride_min": float(config["origin_stride_min_screen"]),
                "train_stride_min": float(stage["train_stride_min"]), "validation_days": int(stage["validation_days"]),
                "min_window_share": float(stage["min_window_share"]),
                "recent_complete_min": float(stage["recent_complete_min"]),
                "window_days": int(config["anchors"]["train_window_days"]),
                "test_days": int(config["anchors"]["max_age_days"]),
                "clarke_point_sample": int(config["evaluation"]["clarke_point_sample"]),
                "seeds": [int(s) for s in cfg["seeds"]], "population_dir": str(population_dir),
                "finetune": {f: (choices.get(f) or {}).get("finetune") for f in FAMILIES},
                "forgetting": float(cfg.get("arx_forgetting", 1.0))}
    for f in FAMILIES:
        settings[f] = (choices.get(f) or {}).get("scratch") or stage.get(f)
    for m in STATISTICAL_MODELS:
        settings[m] = stage.get(m) or {}
    sets = parameter_sets(cfg["insulin_action_min"], cfg["insulin_peak_min"], cfg["carb_absorption_min"])
    default = default_set(features)
    if default not in sets:
        sets.append(default)
    lightgbm = None if args.no_lightgbm else cfg.get("lightgbm")
    notes = []
    if lightgbm and split_name(lightgbm)[0] in FAMILIES and split_name(lightgbm)[1] != "scratch" and not choices:
        notes.append(f"{lightgbm} left out: no population models ({population_dir / 'choices.yaml'})")
        lightgbm = None
    interval, interval_note = (pa.confirmation_intervals(derived, cfg.get("interval_days", "confirmation"),
                                                         [lightgbm]) if lightgbm else ({None: 0}, ""))
    interval = int(interval[lightgbm]) if lightgbm else 0

    patients = group_patients(core, cfg.get("group", "screen"))
    if args.patients:
        patients = patients[: args.patients]
    anchors = ProcessedStore(core, root=derived, stage="clean").read_extra("anchors")
    anchors = anchors[(anchors["cohort"] == "core") & anchors["patient_id"].isin(patients) &
                      (anchors["period"] == period) & anchors["available"].astype(bool) &
                      anchors["anchor_day"].isin([int(a) for a in cfg["anchor_days"]])].copy()
    anchors["anchor"] = pd.to_datetime(anchors["day0"]) + pd.to_timedelta(anchors["anchor_day"], unit="D")
    anchors = anchors.sort_values(["patient_id", "anchor_day"]).reset_index(drop=True)
    runs = derived / "runs" / SCRIPT_ID
    key = cache_key(settings, features, sets, lightgbm, choices, interval)
    recompute = run_start if args.recompute else None
    jobs = []
    for r in anchors.itertuples():
        name = f"{r.patient_id}__{r.period}__{int(r.anchor_day)}"
        jobs.append({"name": name, "dataset": core, "patient_id": r.patient_id, "period": r.period,
                     "anchor_day": int(r.anchor_day), "anchor": r.anchor, "derived_root": derived,
                     "settings": settings, "features": features, "sets": sets, "lightgbm": lightgbm,
                     "interval": interval, "forgetting": settings["forgetting"], "device": "cpu",
                     "stream": len(jobs), "cache": {"path": str(runs / "cache" / f"{name}.pkl"), "key": key,
                                                    "recompute": recompute}})
    print(f"{len(sets)} sets of curves; ARX at every sample (λ = {settings['forgetting']:g})"
          + (f"; {lightgbm} every {interval} days ({interval_note})" if lightgbm else "")
          + f"; {anchors['patient_id'].nunique()} patients")
    statuses = parallel_map(run_job, jobs, resources.n_workers, label="stage-3b patients")

    days, infos, clarke, missing = collect(jobs)
    runs.mkdir(parents=True, exist_ok=True)
    if not days.empty:
        days.to_parquet(runs / "day_metrics.parquet", index=False)
    info = pd.DataFrame(infos)
    is_choice = info["chosen_set"].notna() if "chosen_set" in info else pd.Series(False, index=info.index)
    choices_log = info[is_choice].dropna(axis=1, how="all").reset_index(drop=True)
    fit_log = info[~is_choice].dropna(axis=1, how="all").reset_index(drop=True)
    if not fit_log.empty:
        write_fit_log(fit_log, runs / "fit_log.parquet")
    task_log = pd.DataFrame([{"job": r["job"], "status": r["status"],
                              "retry_errors": " | ".join(r.get("retry_errors", [])), "seconds": r["seconds"],
                              "warnings": r["warnings"],
                              "error": r["error"].strip().splitlines()[-1] if r["error"] else "",
                              "traceback": r["error"]} for r in statuses])

    replicates, seed = int(stage["bootstrap_replicates"]), int(config["random_seed"])
    models = ["arx"] + ([lightgbm] if lightgbm else [])
    dmd = DMDTable(read_noise_floor(Path(out.dir).parent / "12_stage0_population_transfer" / "noise_floor.csv"),
                   [m for m in models if split_name(m)[0] in FAMILIES])
    default_arm = f"fixed_{default.name}"
    skill = pr.patient_skill(days) if not days.empty else pd.DataFrame()
    if not skill.empty:
        skill = pd.concat([skill, oracle_rows(skill, default_arm)], ignore_index=True)
    summary = pr.summary_table(skill, replicates, seed) if not skill.empty else pd.DataFrame()
    fixed = fixed_table(skill, sets, default_arm, dmd, replicates, seed)
    against = {"personal": default_arm, "oracle": default_arm, "restart_personal": "restart", "restart": "never"}
    vs_default = pr.paired(skill, against, dmd, replicates, seed) if not skill.empty else pd.DataFrame()
    oracle = oracle_table(skill, choices_log, default_arm, lightgbm)
    choice_summary = choice_table(oracle, sets)
    overall = finish(days.groupby(["period", "model", "policy", "horizon_min"], as_index=False)[SUMS].sum()).round(4) \
        if not days.empty else pd.DataFrame()
    sample = clarke.sample()
    if not sample.empty:  # the histogram has every panel; the points only those of the main arms
        sample = sample[sample["policy"].isin([default_arm, "personal", "restart", "restart_personal", "reference"])]
    gap = float(fit_log["default_gap"].max()) if "default_gap" in fit_log and fit_log["default_gap"].notna().any() \
        else np.nan

    out.csv(fixed.round(5), "fixed_sets")
    out.csv(summary.round(4), "arm_summary")
    out.csv(vs_default.round(4), "vs_default")
    out.csv(oracle.round(5), "oracle")
    out.csv(choices_log, "choices")
    out.csv(choice_summary, "choice_summary")
    out.csv(overall, "metrics_overall")
    out.csv(clarke.zones(), "clarke_zones")
    out.csv(clarke.histogram(), "clarke_histogram")
    out.csv(sample, "clarke_points_sample")
    out.csv(fit_log, "fit_log")
    out.csv(fit_outcomes(fit_log.drop_duplicates([c for c in FIT_KEYS + ["policy"] if c in fit_log]))
            if not fit_log.empty else pd.DataFrame(), "fit_outcomes")
    out.csv(task_log, "task_log")
    out.csv(anchors.drop(columns=["dataset"], errors="ignore"), "jobs")
    if not fixed.empty and fixed["skill_difference"].notna().any():
        out.figure(plot_fixed(fixed, period), "fixed_sets")
    if not oracle.empty:
        out.figure(plot_choices(oracle, sets, default, period), "choices")
        out.figure(plot_personal(oracle, lightgbm, period), "personal")
    out.text(write_summary(fixed, vs_default, oracle, choice_summary, fit_log, task_log, missing, sets, default,
                           lightgbm, interval, interval_note, notes, gap, anchors, release, resources, settings,
                           period), "summary.md")
    out.manifest(settings={**{k: v for k, v in settings.items() if k != "population_dir"}, "period": period,
                           "sets": [s.name for s in sets], "default_set": default.name, "lightgbm": lightgbm,
                           "interval": interval, "patients": int(anchors["patient_id"].nunique()),
                           "population_models": str(population_dir), "choices_release": choices.get("release")},
                 notes="One job per patient; day metrics in <derived>/runs/42_stage3b_physiology.")
    print(f"Done. Results in {out.dir}")


def collect(jobs: list[dict]):
    frames, infos, missing = [], [], []
    clarke = ClarkeAccumulator(["model", "policy", "horizon_min", "period"], sample_size=20000)
    for job in jobs:
        cached = read_cache(job["cache"])
        if cached is None:
            missing.append(job["name"])
            continue
        frames.append(cached["days"])
        infos += cached["info"]
        if cached["clarke"] is not None:
            clarke.merge(cached["clarke"])
    days = pd.concat([f for f in frames if not f.empty], ignore_index=True) if frames else pd.DataFrame()
    return days, infos, clarke, missing


def write_fit_log(fit_log: pd.DataFrame, path: Path) -> None:
    columns = [c for c in FIT_KEYS + ["policy", "physiology_set", "test_origins", "train_days", "train_samples",
                                      "fit_seconds", "compute_seconds", "rounds", "added_rounds", "default_gap",
                                      "error", "excluded", "served_by"] if c in fit_log]
    table = fit_log[columns].copy()
    for c in ("fit_day", "train_start"):
        if c in table:
            table[c] = pd.to_datetime(table[c].astype(str), errors="coerce")
    for c in table.columns:
        if table[c].dtype == object:
            table[c] = table[c].astype("string")
    table.to_parquet(path, index=False)


# ============================================================================ analysis


def set_parts(name: str) -> dict:
    """{'action_min': 360.0, 'peak_min': 75.0, 'absorption_min': 180.0} for 'dia360_peak75_carb180'."""
    action, rest = name[len("dia"):].split("_peak")
    peak, absorption = rest.split("_carb")
    return {"action_min": float(action), "peak_min": float(peak), "absorption_min": float(absorption)}


def oracle_rows(skill: pd.DataFrame, default_arm: str) -> pd.DataFrame:
    """ARX's oracle arm: for each patient, the fixed set with the best mean skill over horizons on the test days
    (an upper bound for the personal choice, not a method)."""
    fixed = skill[(skill["model"] == "arx") & skill["policy"].str.startswith("fixed_")]
    if fixed.empty:
        return pd.DataFrame()
    mean = fixed.groupby(["period", "patient_id", "policy"])["skill"].mean().reset_index()
    best = mean.loc[mean.groupby(["period", "patient_id"])["skill"].idxmax(), ["period", "patient_id", "policy"]]
    rows = fixed.merge(best, on=["period", "patient_id", "policy"])
    return rows.assign(oracle_set=rows["policy"].str[len("fixed_"):], policy="oracle")


def fixed_table(skill: pd.DataFrame, sets: list, default_arm: str, dmd, replicates: int, seed: int) -> pd.DataFrame:
    """ARX: per set and horizon, the mean skill over patients and the paired difference with the default set."""
    if skill.empty:
        return pd.DataFrame()
    fixed = skill[(skill["model"] == "arx") & skill["policy"].str.startswith("fixed_")]
    mean = fixed.groupby(["period", "model", "policy", "horizon_min"])["skill"].mean().rename("skill_mean")
    paired = pr.paired(fixed, lambda a: default_arm if a != default_arm else None, dmd, replicates, seed)
    table = mean.reset_index()
    if not paired.empty:  # pairs need at least 3 patients
        table = table.merge(paired, on=["period", "model", "policy", "horizon_min"], how="left")
    for column in ("skill_difference", "low", "high", "dmd", "verdict"):
        if column not in table:
            table[column] = np.nan
    parts = pd.DataFrame([set_parts(p[len("fixed_"):]) for p in table["policy"]], index=table.index)
    table = pd.concat([table, parts], axis=1)
    table["default"] = table["policy"] == default_arm
    return table.sort_values(["period", "horizon_min", "action_min", "peak_min", "absorption_min"]).reset_index(
        drop=True)


def oracle_table(skill: pd.DataFrame, choices_log: pd.DataFrame, default_arm: str, lightgbm: str | None
                 ) -> pd.DataFrame:
    """Per patient: the oracle set and the last set chosen from the past, the number of changes of the choice, and
    the gains (mean skill over horizons, points) of the personal choice and of the oracle over the default set
    (ARX), and of restart_personal over restart (LightGBM)."""
    if skill.empty:
        return pd.DataFrame()
    mean = skill.groupby(["period", "model", "policy", "patient_id"])["skill"].mean()
    rows = []
    oracle_sets = skill[skill["policy"] == "oracle"].groupby("patient_id")["oracle_set"].first() \
        if "oracle_set" in skill else pd.Series(dtype=str)
    for (period, patient), _ in skill[skill["model"] == "arx"].groupby(["period", "patient_id"]):
        def get(model, policy):
            return mean.get((period, model, policy, patient), np.nan)
        row = {"period": period, "patient_id": patient, "oracle_set": oracle_sets.get(patient),
               "arx_default": get("arx", default_arm), "arx_personal_gain": get("arx", "personal") -
               get("arx", default_arm), "arx_oracle_gain": get("arx", "oracle") - get("arx", default_arm)}
        if lightgbm:
            row["lightgbm_restart"] = get(lightgbm, "restart")
            row["lightgbm_personal_gain"] = get(lightgbm, "restart_personal") - get(lightgbm, "restart")
        c = choices_log[choices_log["patient_id"] == patient].sort_values("choice_day") \
            if not choices_log.empty else pd.DataFrame()
        if not c.empty:
            row["last_choice"] = c["chosen_set"].iloc[-1]
            row["changes"] = int((c["chosen_set"] != c["chosen_set"].shift()).sum() - 1)
            row["days_on_default"] = int((c["chosen_set"] == default_arm[len("fixed_"):]).sum())
            row["choice_days"] = int(len(c))
        rows.append(row)
    table = pd.DataFrame(rows)
    if "last_choice" in table:
        table["last_is_oracle"] = table["last_choice"] == table["oracle_set"]
    return table


def choice_table(oracle: pd.DataFrame, sets: list) -> pd.DataFrame:
    """Per set: patients whose last choice from the past it is, and patients whose oracle it is."""
    if oracle.empty:
        return pd.DataFrame()
    rows = []
    for s in sets:
        rows.append({"set": s.name, "action_min": s.action_min, "peak_min": s.peak_min,
                     "absorption_min": s.absorption_min,
                     "last_choice_patients": int((oracle.get("last_choice") == s.name).sum())
                     if "last_choice" in oracle else 0,
                     "oracle_patients": int((oracle["oracle_set"] == s.name).sum())})
    return pd.DataFrame(rows)


# ============================================================================ figures


def plot_fixed(fixed: pd.DataFrame, period: str):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    f = fixed[fixed["period"] == period].copy()
    f["row"] = [f"{a:g} min" + (f" (peak {p:g})" if f["peak_min"].nunique() > 1 else "")
                for a, p in zip(f["action_min"], f["peak_min"])]
    horizons = sorted(f["horizon_min"].unique())
    fig, axes = plt.subplots(1, len(horizons) + 1, figsize=(3.4 * len(horizons) + 0.9, 3.4), squeeze=False,
                             gridspec_kw={"width_ratios": [1] * len(horizons) + [0.06]})
    bar_ax, axes = axes[0][-1], axes[:, :-1]
    cmap = LinearSegmentedColormap.from_list("diverging", DIVERGING)
    values = 100 * f["skill_difference"].fillna(0.0)
    span = max(0.1, float(values.abs().max()))
    for ax, horizon in zip(axes[0], horizons):
        h = f[f["horizon_min"] == horizon]
        grid = h.pivot_table(index="row", columns="absorption_min", values="skill_difference", aggfunc="first",
                             sort=False).fillna(0.0) * 100
        rows = sorted(grid.index, key=lambda r: float(r.split()[0]))
        grid = grid.loc[rows]
        image = ax.imshow(grid.to_numpy(), cmap=cmap, norm=TwoSlopeNorm(0.0, -span, span), aspect="auto")
        verdict = h.set_index(["row", "absorption_min"])["verdict"]
        is_default = h.set_index(["row", "absorption_min"])["default"]
        for i, r in enumerate(grid.index):
            for j, c in enumerate(grid.columns):
                text = "default" if is_default.get((r, c), False) else f"{grid.iloc[i, j]:+.2f}"
                weight = "bold" if verdict.get((r, c)) in ("better", "worse") else "normal"
                ax.text(j, i, text, ha="center", va="center", fontsize=7, fontweight=weight, color="#0b0b0b")
        ax.set_xticks(range(len(grid.columns)), [f"{c:g}" for c in grid.columns], fontsize=7)
        ax.set_yticks(range(len(grid.index)), grid.index if ax is axes[0][0] else [""] * len(grid.index), fontsize=7)
        ax.grid(False)
        ax.set_xlabel("carbohydrate absorption (min)")
        ax.set_title(f"{int(horizon)} min", loc="left", fontsize=9)
    axes[0][0].set_ylabel("insulin action duration")
    fig.colorbar(image, cax=bar_ax, label="skill points against the default set")
    fig.suptitle(f"ARX with other insulin and carbohydrate curves for everyone (bold: beyond the DMD), "
                 f"{PERIOD_LABELS.get(period, period)}", fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    return fig


def plot_choices(oracle: pd.DataFrame, sets: list, default, period: str):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    o = oracle[oracle["period"] == period]
    names = [s.name for s in sets]
    short = [f"{s.action_min:g}/{s.absorption_min:g}" + (f"/{s.peak_min:g}" if len({x.peak_min for x in sets}) > 1
                                                          else "") for s in sets]
    x = np.arange(len(names))
    last = [int((o.get("last_choice") == n).sum()) if "last_choice" in o else 0 for n in names]
    best = [int((o["oracle_set"] == n).sum()) for n in names]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.4), gridspec_kw={"width_ratios": [3, 1.2]})
    ax = axes[0]
    ax.bar(x - 0.2, last, 0.4, color="#2a78d6", label="chosen from the past (last test day)")
    ax.bar(x + 0.2, best, 0.4, color="#9b9a96", label="best on the test days (oracle)")
    ax.set_xticks(x, short, fontsize=7, rotation=45, ha="right")
    ax.get_xticklabels()[names.index(default.name)].set_fontweight("bold")
    ax.set_xlabel("insulin action / carbohydrate absorption (min); default in bold")
    ax.set_ylabel("patients")
    ax.legend(fontsize=7)
    ax.set_title("Curves of each patient", loc="left", fontsize=9)
    ax = axes[1]
    if "changes" in o:
        counts = o["changes"].value_counts().sort_index()
        ax.bar(counts.index.astype(int), counts.to_numpy(), color="#2a78d6", width=0.8)
    ax.set_xlabel("changes of the choice over the test days")
    ax.set_ylabel("patients")
    for a in axes:
        a.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_title("Stability of the choice", loc="left", fontsize=9)
    fig.suptitle(f"Stage 3b: the curves chosen for each patient, {PERIOD_LABELS.get(period, period)}",
                 fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    return fig


def plot_personal(oracle: pd.DataFrame, lightgbm: str | None, period: str):
    import matplotlib.pyplot as plt

    o = oracle[oracle["period"] == period].sort_values("arx_personal_gain").reset_index(drop=True)
    panels = [("arx_personal_gain", "arx_oracle_gain", "ARX: personal curves against the default")]
    if lightgbm and "lightgbm_personal_gain" in o:
        panels.append(("lightgbm_personal_gain", None, f"{label(lightgbm)}: restart with the personal curves"))
    fig, axes = plt.subplots(1, len(panels), figsize=(4.8 * len(panels), 3.6), squeeze=False)
    y = np.arange(len(o))
    for ax, (column, extra, title) in zip(axes[0], panels):
        ax.scatter(100 * o[column], y, s=16, color="#2a78d6", label="chosen from the past", zorder=3)
        if extra:
            ax.scatter(100 * o[extra], y, s=16, marker="D", facecolors="white", edgecolors="#9b9a96",
                       label="oracle (upper bound)", zorder=2)
        ax.axvline(0, color=TEXT_SECONDARY, lw=0.8)
        ax.set_yticks([])
        ax.set_ylabel("patients (sorted by ARX's gain)")
        ax.set_xlabel("gain in skill points (mean of horizons)")
        ax.set_title(title, loc="left", fontsize=9)
        ax.legend(fontsize=7)
    fig.suptitle(f"Stage 3b: gain of each patient's own curves, {PERIOD_LABELS.get(period, period)}",
                 fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    return fig


# ============================================================================ summary


def write_summary(fixed, vs_default, oracle, choice_summary, fit_log, task_log, missing, sets, default, lightgbm,
                  interval, interval_note, notes, gap, anchors, release, resources, settings, period) -> str:
    lines = ["# 42 - Stage 3b: insulin action duration and carbohydrate absorption", "",
             f"Code release {release}.", f"Machine: {resources.describe()}",
             f"{len(sets)} sets of curves (insulin action x peak x carbohydrate absorption, minutes); default "
             f"{default.name} (script 05, D54). ARX updated at every sample, λ = {settings['forgetting']:g} (D87)"
             + (f"; {lightgbm} restarted from the population model every {interval} days ({interval_note})"
                if lightgbm else "; LightGBM left out")
             + f". {anchors['patient_id'].nunique()} patients of the screen group, anchor day(s) "
             f"{sorted(anchors['anchor_day'].unique().tolist())}, {PERIOD_LABELS.get(period, period)}.", ""]
    problems = list(notes)
    if not task_log.empty:
        problems += [f"{r.job} failed: {r.error}" for r in task_log[task_log["status"] == "error"].itertuples()]
        problems += [f"{r.job} raised warnings: {str(r.warnings)[:200]}"
                     for r in task_log[task_log["warnings"].fillna("") != ""].head(10).itertuples()]
        retry = task_log[task_log["retry_errors"].fillna("") != ""]
        problems += [f"{r.job}: fits failed, not cached (computed again next run): {str(r.retry_errors)[:300]}"
                     for r in retry.head(10).itertuples()]
    problems += [f"no result for {m}" for m in missing[:10]]
    if np.isfinite(gap) and gap > 1e-3:
        problems.append(f"the default curves recomputed here differ from the saved grid (script 05) by up to {gap:.4g}"
                        " (inputs in their units): run script 05 again")
    if not fit_log.empty and "excluded" in fit_log:
        out = fit_log[fit_log["excluded"].notna()].drop_duplicates(["model", "patient_id"])
        for (model, reason), g in out.groupby(["model", "excluded"]):
            problems.append(f"{model}: left out for {', '.join(g['patient_id'])} ({reason})")
    if not fit_log.empty and "error" in fit_log:
        kept = fit_log["excluded"].isna() if "excluded" in fit_log else pd.Series(True, index=fit_log.index)
        failed = fit_log[fit_log["error"].notna() & kept]
        for (model, error), g in failed.groupby(["model", "error"]):
            problems.append(f"{model}: {len(g)} fits failed and kept the model in use: {error}")
    lines += ["## Warnings", ""] + ([f"- {p}" for p in problems] or ["- none"]) + [""]
    if not fit_log.empty:
        outcomes = fit_outcomes(fit_log.drop_duplicates([c for c in FIT_KEYS + ["policy"] if c in fit_log]))
        lines += ["## Fits (a run is complete when `other_failure` is 0)", "", markdown_table(outcomes), "",
                  FIT_NOTE, ""]
    if np.isfinite(gap):
        lines += [f"Default curves recomputed against the saved grid: largest difference {gap:.2g}.", ""]
    if not fixed.empty:
        f = fixed[fixed["period"] == period].copy()
        f["cell"] = [("default" if d else f"{100 * x:+.2f} ({v})") for d, x, v in
                     zip(f["default"], f["skill_difference"], f["verdict"])]
        wide = f.pivot_table(index=["action_min", "peak_min", "absorption_min"], columns="horizon_min",
                             values="cell", aggfunc="first")
        wide.columns = [f"{int(c)} min" for c in wide.columns]
        best = f[~f["default"]].groupby("policy")["skill_difference"].mean().sort_values(ascending=False)
        lines += ["## ARX: each set for everyone against the default (skill points, verdict with the DMD)", "",
                  markdown_table(wide.reset_index()), ""]
        if not best.empty:
            lines += [f"Best set for everyone, mean over horizons: {best.index[0][len('fixed_'):]} "
                      f"({100 * best.iloc[0]:+.2f} points; chosen on these same patients, so optimistic).", ""]
    if not vs_default.empty:
        v = vs_default[vs_default["period"] == period].copy()
        v["cell"] = [f"{100 * d:+.2f} ({100 * lo:+.2f} to {100 * hi:+.2f}; {verdict})" for d, lo, hi, verdict in
                     zip(v["skill_difference"], v["low"], v["high"], v["verdict"])]
        wide = v.pivot_table(index=["model", "policy", "against"], columns="horizon_min", values="cell",
                             aggfunc="first")
        wide.columns = [f"{int(c)} min" for c in wide.columns]
        lines += ["## Each patient's own curves (skill points, 95 % CI, verdict with the model's DMD)", "",
                  "`personal`: the set with the best forecasts so far, chosen every day from the patient's past; "
                  "`oracle`: the best set on the test days (an upper bound, not a method); `restart_personal`: "
                  "LightGBM refitted with ARX's choice of the day.", "", markdown_table(wide.reset_index()), ""]
    if not oracle.empty:
        o = oracle[oracle["period"] == period]
        facts = []
        if "last_is_oracle" in o:
            facts.append(f"the last choice from the past is the oracle set for {int(o['last_is_oracle'].sum())} of "
                         f"{len(o)} patients")
        if "changes" in o:
            facts.append(f"the choice changes a median of {o['changes'].median():g} times over the test days "
                         f"(maximum {int(o['changes'].max())})")
        if "days_on_default" in o:
            share = o["days_on_default"].sum() / max(1, o["choice_days"].sum())
            facts.append(f"the default set is chosen on {100 * share:.1f} % of the patient-days")
        lines += ["## Choices", "", "; ".join(facts) + ".", "", markdown_table(
            choice_summary[(choice_summary["last_choice_patients"] > 0) | (choice_summary["oracle_patients"] > 0)]),
            ""]
    lines += ["## Checks to look at", "",
              "- `fixed_sets.png`: whether the default curves are a good choice for everyone.",
              "- `choices.png`: whether patients differ in their curves, and how stable the choice is.",
              "- `personal.png`: which patients gain from their own curves, and how close the choice from the past "
              "comes to the oracle.", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
