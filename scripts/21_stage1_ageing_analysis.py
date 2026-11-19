"""21 - Stage 1 analysis: ageing slopes, useful life, version effects and screening 1.

Reads the day metrics of script 20 (every patient, anchor, model, version,
horizon and test day of the screen group) and the noise floor of script 12,
and answers Q1 with mixed models (HANDOFF 7.3, D78; ``eval.ageing_analysis``):

- daily skill against persistence on the same origins (days with fewer than
  ``min_origins_per_day`` origins left out, clipped to ``skill_clip``);
- ageing: per model and horizon, the frozen daily skill over the weeks since
  the fit, with random effects of the patient (intercept and slope) and of the
  anchor within the patient; the fixed slope is the skill lost per week;
- useful life: days until that loss reaches the model's DMD (D75), with its
  interval; a model ages within the follow-up when the slope is below zero
  with its interval and the useful life is at most ``anchors.max_age_days``;
- version effects with patient and anchor random intercepts: recency (rolling
  - frozen), amount of data (growing - rolling) and both, with the tie rule;
- screening 1: per mechanism, the ``per_mechanism`` best models in their
  better fresh version, plus models that age clearly more slowly than the
  best of their mechanism; models that never beat persistence are dropped;
- decision point of 7.3: whether any model ages beyond its DMD within the
  follow-up (if none, Q2 is answered "almost never").

Reports RMSE and Clarke zones of every model and version (rule R6; the grid
data are in script 20's results). The mixed models of every (model,
horizon) run in parallel on all cores but two (rule R5). CPU only.

Derived data:
    <derived_data>/runs/21_stage1_ageing_analysis/daily_skill.parquet
    <derived_data>/runs/21_stage1_ageing_analysis/shortlist.yaml   the models that pass screening 1 (read by script 30)

Outputs (results/21_stage1_ageing_analysis/):
    ageing_slopes.csv        per period, model and horizon: slope per week, interval, p, useful life, random part used,
                             and the within-anchor slope (patient-clustered errors) as a check
    version_effects_mixed.csv  per period, model, horizon and effect: difference, interval, DMD, verdict
    screening.csv            per period and model: mechanism, best fresh version and skill, slower ageing, decision
                             (full models only; each glucose-only twin follows its full model, D90)
    glucose_only.csv         D90: per model and horizon, the full model against its glucose-only twin (fresh skill,
                             ageing slope, useful life)
    shortlist.yaml           copy of the shortlist for stage 2
    weekly_daily_skill.csv   observed daily skill by week since the anchor, per model, version and horizon
    metrics_by_version.csv   per period, model, version and horizon: n, RMSE, MAE, MARD, bias, Clarke zones (R6)
    fit_log.csv              every mixed model: rows, patients, random part, notes, seconds
    ageing_fits.png
    ageing_slopes.png
    version_effects_mixed.png
    screening.png
    summary.md
    run_manifest.json

Usage:
    python scripts/21_stage1_ageing_analysis.py
    python scripts/21_stage1_ageing_analysis.py --models ar3 lightgbm_ft     # a subset of the models of script 20
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

from t1dretrain.compute import detect_resources, parallel_map  # noqa: E402
from t1dretrain.config import CONFIG_DIR, derived_data_dir, load_project_config  # noqa: E402
from t1dretrain.data.pipeline import WarningRecorder  # noqa: E402
from t1dretrain.eval import ageing_analysis as aa  # noqa: E402
from t1dretrain.eval.stage0_report import (PERIOD_LABELS, SUMS, DMDTable, color, finish, label,  # noqa: E402
                                           read_noise_floor)
from t1dretrain.outputs import ScriptOutput, markdown_table  # noqa: E402
from t1dretrain.plotting import TEXT_SECONDARY, set_style  # noqa: E402
from t1dretrain.sim.stage0 import (FAMILIES, mechanism_of, split_name, split_variant, twin_families,  # noqa: E402
                                   with_twins)

SCRIPT_ID = "21_stage1_ageing_analysis"
VERSION_COLORS = {"frozen": "#d1495b", "rolling": "#2a78d6", "growing": "#1baf7a"}
MECHANISM_LABELS = {"re_estimation": "re-estimation", "sample_wise": "update at every sample",
                    "continue_training": "continued training (trees)", "global_finetune": "fine-tuned global network",
                    "context_only": "context only"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--paths", type=Path, default=CONFIG_DIR / "paths.yaml", help="paths file (config/paths.yaml)")
    parser.add_argument("--periods", nargs="*", default=None, help="periods (default: stage1.periods)")
    parser.add_argument("--models", nargs="*", default=None, help="subset of the models of script 20")
    parser.add_argument("--skip-code-check", action="store_true",
                        help="run even if the code differs from the released manifest")
    return parser.parse_args()


# ============================================================================ jobs (module level for the workers)


def run_fit(job: dict) -> dict:
    """The ageing and version models of one period, model and horizon."""
    with WarningRecorder() as recorder:
        frame = pd.read_parquet(job["path"], filters=[("period", "==", job["period"]), ("model", "==", job["model"]),
                                                      ("horizon_min", "==", job["horizon_min"])])
        key = {"period": job["period"], "model": job["model"], "horizon_min": job["horizon_min"]}
        try:
            ageing = {**key, **aa.ageing_fit(frame)}
            versions = [{**key, **row} for row in aa.version_fit(frame)]
            error = ""
        except Exception as exc:  # noqa: BLE001
            ageing, versions, error = {**key, "note": f"{type(exc).__name__}: {exc}"[:300]}, [], str(exc)
    return {"ageing": ageing, "versions": versions, "warnings": recorder.text()[:300], "error": error}


# ============================================================================ main


def main() -> None:
    args = parse_args()
    release = verify_code(strict=not args.skip_code_check)
    config = load_project_config()
    resources = detect_resources(config.get("compute", {}).get("reserve_cores"))
    print(resources.describe())
    out = ScriptOutput(SCRIPT_ID, resources=resources)
    set_style()
    derived = derived_data_dir(args.paths)
    cfg, stage1 = config.get("stage1_analysis", {}), config["stage1"]
    follow_up = int(config["anchors"]["max_age_days"])
    min_origins, clip = int(cfg.get("min_origins_per_day", 12)), tuple(cfg.get("skill_clip", [-1.0, 1.0]))
    per_mechanism = int(cfg.get("per_mechanism", 2))
    source = derived / "runs" / "20_stage1_ageing_runs" / "day_metrics.parquet"
    if not source.exists():
        raise SystemExit(f"{source} not found: run scripts/20_stage1_ageing_runs.py first")
    days = pd.read_parquet(source)
    periods = args.periods or list(stage1["periods"])
    days = days[days["period"].isin(periods)]
    models = [m for m in with_twins(stage1["models"], twin_families(config))  # each with its twin (D90)
              if m in set(days["model"]) and (not args.models or m in args.models
                                              or split_variant(m)[0] in args.models)]
    full = [m for m in models if not split_variant(m)[1]]  # screening 1 is made on the full models (D90)
    days = days[days["model"].isin(models + ["persistence"])]
    if days.empty or not models:
        raise SystemExit("no day metrics of the chosen models and periods in script 20's runs")
    print(f"{len(days)} day rows of {days['patient_id'].nunique()} patients; models {models}; periods {periods}")

    skill = aa.daily_skill(days, min_origins, clip)
    runs = derived / "runs" / SCRIPT_ID
    runs.mkdir(parents=True, exist_ok=True)
    skill_path = runs / "daily_skill.parquet"
    skill.to_parquet(skill_path, index=False)
    horizons = sorted(int(h) for h in skill["horizon_min"].unique())
    jobs = [{"path": str(skill_path), "period": p, "model": m, "horizon_min": h}
            for p in periods for m in models for h in horizons]
    results = parallel_map(run_fit, jobs, resources.n_workers, label="mixed models")

    learned = [m for m in models if split_name(m)[0] in FAMILIES]
    dmd = DMDTable(read_noise_floor(Path(out.dir).parent / "12_stage0_population_transfer" / "noise_floor.csv"),
                   learned)
    slopes = pd.DataFrame([r["ageing"] for r in results])
    if not slopes.empty and "slope_per_week" in slopes:
        slopes["dmd"] = [dmd.model(r.period, r.horizon_min, r.model) for r in slopes.itertuples()]
        life = [aa.useful_life(r.slope_per_week, r.slope_low, r.slope_high, r.dmd, follow_up)
                for r in slopes.itertuples()]
        slopes = pd.concat([slopes, pd.DataFrame(life)], axis=1)
    effects = pd.DataFrame([row for r in results for row in r["versions"]])
    if not effects.empty and "difference" in effects:
        effects["dmd"] = [dmd.model(r.period, r.horizon_min, r.model) for r in effects.itertuples()]
        effects["verdict"] = [aa.verdict(r.difference, r.low, r.high, r.dmd) if np.isfinite(r.difference)
                              else "not estimated" for r in effects.itertuples()]
    fresh = aa.fresh_skill(days)
    mechanisms = {m: mechanism_of(m) for m in full}
    full_slopes = slopes[slopes["model"].isin(full)] if not slopes.empty else slopes
    screens = pd.concat([aa.screening(fresh[(fresh["period"] == p) & fresh["model"].isin(full)],
                                      full_slopes[full_slopes["period"] == p] if not full_slopes.empty
                                      else full_slopes, mechanisms, per_mechanism).assign(period=p) for p in periods],
                        ignore_index=True)
    twins = glucose_only_table(fresh, slopes, screens, models)
    weekly = aa.weekly_curve(skill)
    metrics = finish(days.groupby(["period", "model", "version", "horizon_min"], as_index=False)[SUMS].sum()).round(4)
    fit_log = pd.DataFrame([{**{k: r["ageing"].get(k) for k in ("period", "model", "horizon_min", "days", "patients",
                                                                  "anchors", "spec", "note", "seconds")},
                             "version_spec": (r["versions"][0].get("spec") if r["versions"] else ""),
                             "warnings": r["warnings"], "error": r["error"]} for r in results])

    shortlist = {"release": release, "periods": {p: {
        "models": screens.loc[(screens["period"] == p) & screens["passes"], "model"].tolist(),
        "fresh_version": dict(zip(screens.loc[screens["period"] == p, "model"],
                                  screens.loc[screens["period"] == p, "best_fresh_version"]))}
        for p in periods} if not screens.empty else {},
        "note": "screening 1 of HANDOFF 7.3 (D78); foundation models go to stage 3 directly (D70)"}
    (runs / "shortlist.yaml").write_text(yaml.safe_dump(shortlist, sort_keys=False))
    out.text(yaml.safe_dump(shortlist, sort_keys=False), "shortlist.yaml")
    out.csv(slopes.round(5), "ageing_slopes")
    out.csv(effects.round(5), "version_effects_mixed")
    out.csv(screens.round(4), "screening")
    out.csv(twins.round(5), "glucose_only")
    out.csv(weekly.round(4), "weekly_daily_skill")
    out.csv(metrics, "metrics_by_version")
    out.csv(fit_log, "fit_log")
    period = periods[0]
    if not slopes.empty and "slope_per_week" in slopes:  # the full models (their twins: glucose_only.csv)
        out.figure(plot_fits(weekly, slopes, full, period), "ageing_fits")
        out.figure(plot_slopes(slopes, full, period), "ageing_slopes")
    if not effects.empty and "difference" in effects:
        out.figure(plot_effects(effects, full, period), "version_effects_mixed")
    if not screens.empty:
        out.figure(plot_screening(screens, slopes, period, list(config.get("stage2", {}).get("extra_models", []))),
                   "screening")
    out.text(write_summary(slopes, effects, screens, fit_log, skill, models, periods, follow_up, min_origins, clip,
                           per_mechanism, len(dmd), release, resources, twins), "summary.md")
    out.manifest(settings={"periods": periods, "models": models, "min_origins_per_day": min_origins,
                           "skill_clip": list(clip), "per_mechanism": per_mechanism, "follow_up_days": follow_up},
                 notes="Daily skill in <derived>/runs/21_stage1_ageing_analysis/daily_skill.parquet.")
    print(f"Done. Results in {out.dir}")


# ============================================================================ figures


def plot_fits(weekly: pd.DataFrame, slopes: pd.DataFrame, models: list[str], period: str) -> plt.Figure:
    w = weekly[(weekly["period"] == period) & (weekly["version"] == "frozen")]
    horizons = sorted(w["horizon_min"].unique())
    fig, axes = plt.subplots(len(models), len(horizons), figsize=(4.6 * len(horizons), 2.8 * len(models)),
                             squeeze=False, sharex=True)
    for i, model in enumerate(models):
        for j, horizon in enumerate(horizons):
            ax = axes[i, j]
            v = w[(w["model"] == model) & (w["horizon_min"] == horizon)]
            ax.fill_between(v["week"], v["skill_low"], v["skill_high"], color=color(model), alpha=0.18, lw=0)
            ax.plot(v["week"], v["skill_mean"], "o", color=color(model), ms=3.5, label="observed (weekly mean)")
            s = slopes[(slopes["period"] == period) & (slopes["model"] == model) & (slopes["horizon_min"] == horizon)]
            if not s.empty and np.isfinite(s["slope_per_week"].iloc[0]):
                r = s.iloc[0]
                x = np.linspace(0, max(1.0, v["week"].max() + 1 if len(v) else 12), 20)
                ax.plot(x, r["intercept"] + r["slope_per_week"] * x, color=TEXT_SECONDARY, lw=1.4,
                        label="mixed model")
                ax.set_title(f"{label(model)}, {horizon} min: {100 * r['slope_per_week']:+.2f} points/week",
                             loc="left", fontsize=9)
            else:
                ax.set_title(f"{label(model)}, {horizon} min", loc="left", fontsize=9)
            if i == len(models) - 1:
                ax.set_xlabel("weeks since the fit")
        axes[i, 0].set_ylabel("daily skill")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(f"Frozen models: daily skill over the weeks without refitting, {PERIOD_LABELS.get(period, period)}",
                 fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    return fig


def plot_slopes(slopes: pd.DataFrame, models: list[str], period: str) -> plt.Figure:
    s = slopes[slopes["period"] == period]
    horizons = sorted(s["horizon_min"].unique())
    fig, axes = plt.subplots(1, len(horizons), figsize=(4.8 * len(horizons), 0.45 * len(models) + 1.6), sharey=True,
                             squeeze=False)
    y = np.arange(len(models))[::-1]
    for ax, horizon in zip(axes[0], horizons):
        h = s[s["horizon_min"] == horizon].set_index("model").reindex(models)
        for yy, (model, r) in zip(y, h.iterrows()):
            if not np.isfinite(r.get("slope_per_week", np.nan)):
                continue
            if np.isfinite(r["dmd"]):  # the loss that reaches the DMD within the follow-up, per week
                ax.fill_between([-100 * r["dmd"] / 12, 0], yy - 0.35, yy + 0.35, color=TEXT_SECONDARY, alpha=0.12,
                                lw=0, zorder=0)
            ax.plot([100 * r["slope_low"], 100 * r["slope_high"]], [yy, yy], color=color(model), lw=2)
            ax.scatter(100 * r["slope_per_week"], yy, color=color(model), s=30, zorder=3,
                       marker="o" if r["ages_within_follow_up"] else "s")
        ax.axvline(0, color=TEXT_SECONDARY, lw=0.8)
        ax.set_yticks(y, [label(m) for m in models], fontsize=8)
        ax.set_title(f"{horizon} min", loc="left", fontsize=10)
        ax.set_xlabel("skill points per week (95 % CI)")
    fig.suptitle("Ageing of the frozen models (circle: reaches its DMD within the follow-up; band: a loss smaller "
                 "than the DMD over 12 weeks)", fontweight="bold", x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    return fig


def plot_effects(effects: pd.DataFrame, models: list[str], period: str) -> plt.Figure:
    e = effects[(effects["period"] == period) & effects["effect"].isin(list(aa.EFFECTS)[:2])].copy()
    e["row"] = e["model"].map(label) + ", " + e["effect"].str.split(" ").str[0]
    rows = [f"{label(m)}, {k}" for m in models for k in ("recency", "data")]
    horizons = sorted(e["horizon_min"].unique())
    fig, axes = plt.subplots(1, len(horizons), figsize=(5.0 * len(horizons), 0.36 * len(rows) + 1.6), sharey=True,
                             squeeze=False)
    y = np.arange(len(rows))[::-1]
    for ax, horizon in zip(axes[0], horizons):
        h = e[e["horizon_min"] == horizon].set_index("row").reindex(rows)
        for yy, (_, r) in zip(y, h.iterrows()):
            if not np.isfinite(r.get("difference", np.nan)):
                continue
            if np.isfinite(r["dmd"]):
                ax.fill_between([-100 * r["dmd"], 100 * r["dmd"]], yy - 0.35, yy + 0.35, color=TEXT_SECONDARY,
                                alpha=0.12, lw=0, zorder=0)
            colour = VERSION_COLORS["rolling" if r["effect"].startswith("recency") else "growing"]
            ax.plot([100 * r["low"], 100 * r["high"]], [yy, yy], color=colour, lw=2)
            ax.scatter(100 * r["difference"], yy, color=colour, s=26, zorder=3,
                       marker="o" if r["verdict"] in ("positive", "negative") else "s")
        ax.axvline(0, color=TEXT_SECONDARY, lw=0.8)
        ax.set_yticks(y, rows, fontsize=7)
        ax.set_title(f"{horizon} min", loc="left", fontsize=10)
        ax.set_xlabel("difference in daily skill, points (95 % CI)")
    fig.suptitle("What refitting adds (mixed models; blue: recency, rolling - frozen; green: data, growing - "
                 "rolling; band: the model's DMD)", fontweight="bold", x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    return fig


def plot_screening(screens: pd.DataFrame, slopes: pd.DataFrame, period: str, extra: list[str]) -> plt.Figure:
    s = screens[screens["period"] == period]
    mean_slope = slopes[slopes["period"] == period].groupby("model")["slope_per_week"].mean() \
        if not slopes.empty and "slope_per_week" in slopes else pd.Series(dtype=float)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for _, r in s.iterrows():
        x = 100 * mean_slope.get(r["model"], np.nan)
        y = 100 * r["fresh_skill_mean"]
        ax.scatter(x, y, s=70, color=color(r["model"]), edgecolor="black" if r["passes"] else "none", lw=1.4,
                   marker="o" if r["passes"] else "X", zorder=3)
        added = "\nadded to stage 2 by decision (D79)" if r["model"] in extra and not r["passes"] else ""
        ax.annotate(f"{label(r['model'])}\n({MECHANISM_LABELS.get(r['mechanism'], r['mechanism'])}){added}",
                    (x, y), textcoords="offset points", xytext=(6, 4), fontsize=7)
    ax.axvline(0, color=TEXT_SECONDARY, lw=0.8)
    ax.set_xlabel("frozen ageing, mean over horizons (skill points per week)")
    ax.set_ylabel("best fresh version, mean skill over horizons (%)")
    ax.set_title(f"Screening 1, {PERIOD_LABELS.get(period, period)} (circle with edge: passes)", loc="left",
                 fontweight="bold")
    fig.tight_layout()
    return fig


# ============================================================================ summary


def glucose_only_table(fresh: pd.DataFrame, slopes: pd.DataFrame, screens: pd.DataFrame, models: list[str]
                       ) -> pd.DataFrame:
    """D90: per period, full model and horizon, the full model against its glucose-only twin: skill of the full
    model's better fresh version, ageing slope per week (with interval) and useful life of the frozen models."""
    rows = []
    best = dict(zip(zip(screens["period"], screens["model"]), screens["best_fresh_version"])) \
        if not screens.empty else {}
    f = fresh.set_index(["period", "model", "version", "horizon_min"])["skill"] if not fresh.empty \
        else pd.Series(dtype=float)
    sl = slopes.set_index(["period", "model", "horizon_min"]) if not slopes.empty and "slope_per_week" in slopes \
        else pd.DataFrame()
    for twin in [m for m in models if split_variant(m)[1]]:
        model = split_variant(twin)[0]
        if model not in models:
            continue
        for (period, horizon) in sorted({(p, h) for p, m, _, h in f.index if m == model}):
            version = best.get((period, model), "growing")
            row = {"period": period, "model": model, "twin": twin, "horizon_min": int(horizon),
                   "fresh_version": version, "fresh_skill": f.get((period, model, version, horizon), np.nan),
                   "fresh_skill_twin": f.get((period, twin, version, horizon), np.nan)}
            row["fresh_difference"] = row["fresh_skill"] - row["fresh_skill_twin"]
            for name, suffix in ((model, ""), (twin, "_twin")):
                if (period, name, horizon) in sl.index:
                    r = sl.loc[(period, name, horizon)]
                    row.update({f"slope_per_week{suffix}": r.get("slope_per_week"), f"slope_low{suffix}": r.get(
                        "slope_low"), f"slope_high{suffix}": r.get("slope_high"),
                        f"useful_life_days{suffix}": r.get("useful_life_days")})
            rows.append(row)
    return pd.DataFrame(rows)


def write_summary(slopes, effects, screens, fit_log, skill, models, periods, follow_up, min_origins, clip,
                  per_mechanism, n_dmd, release, resources, twins=None) -> str:
    lines = ["# 21 - Stage 1 analysis: ageing, useful life and screening 1", "", f"Code release {release}.",
             f"Machine: {resources.describe()}",
             f"Models {models}; periods {periods}; daily skill against persistence on days with at least {min_origins} "
             f"origins, clipped to {list(clip)} ({int(skill['clipped'].sum())} of {len(skill)} model-days clipped); "
             f"follow-up {follow_up} days; DMD from script 12 ({'found' if n_dmd else 'NOT found: no ties'}).", ""]
    problems = [f"{r.model} {r.horizon_min} min: {r.note}" for r in fit_log.itertuples()  # zero variances: fit_log
                if isinstance(r.note, str) and ("not converged" in r.note or "too few" in r.note)]
    problems += [f"{r.model} {r.horizon_min} min raised warnings: {r.warnings}" for r in fit_log.itertuples()
                 if isinstance(r.warnings, str) and r.warnings]
    problems += [f"{r.model} {r.horizon_min} min failed: {r.error}" for r in fit_log.itertuples() if r.error]
    lines += ["## Warnings", ""] + ([f"- {p}" for p in problems[:20]] or ["- none"]) + [""]
    for period in periods:
        s = slopes[slopes["period"] == period] if not slopes.empty else slopes
        lines += [f"## Decision point of 7.3, {PERIOD_LABELS.get(period, period)}", ""]
        if s.empty or "ages_within_follow_up" not in s:
            lines += ["Not estimable: no ageing model could be fitted (see the warnings).", ""]
        else:
            ageing = s[s["ages_within_follow_up"] == True]  # noqa: E712
            if ageing.empty:
                lines += ["No model loses more than its DMD within the follow-up: Q2 is answered \"almost never\" and "
                          "the weight of the paper moves to Q3 (HANDOFF 7.3).", ""]
            else:
                found = ", ".join(f"{label(r.model)} at {r.horizon_min} min ({r.useful_life_days:.0f} days)"
                                  for r in ageing.itertuples())
                lines += [f"Models that lose more than their DMD within {follow_up} days without refitting: {found}. "
                          "Q2 (how often and with which window) stays open.", ""]
            show = ["model", "horizon_min", "days", "patients", "slope_per_week", "slope_low", "slope_high", "slope_p",
                    "within_slope", "dmd", "useful_life_days", "useful_life_low", "useful_life_high",
                    "ages_within_follow_up", "spec"]
            lines += ["## Ageing of the frozen models (skill lost per week; useful life in days)", "",
                      markdown_table(s[[c for c in show if c in s.columns]].round(4)), ""]
        e = effects[effects["period"] == period] if not effects.empty and "period" in effects else pd.DataFrame()
        if not e.empty and "difference" in e:
            show = ["model", "horizon_min", "effect", "difference", "low", "high", "dmd", "verdict", "days", "spec"]
            lines += ["## Version effects (mixed models of the daily skill; tie = below the model's DMD)", "",
                      markdown_table(e[[c for c in show if c in e.columns]].round(4)), ""]
        sc = screens[screens["period"] == period] if not screens.empty else screens
        if not sc.empty:
            show = [c for c in sc.columns if c not in ("period", "passes")]
            lines += [f"## Screening 1 (best {per_mechanism} per mechanism in the better fresh version, plus slower "
                      "ageing; dropped if never better than persistence)", "", markdown_table(sc[show]), "",
                      "Foundation models (context only) are not refitted and go to stage 3 directly (D70).", ""]
    if twins is not None and not twins.empty:
        show = [c for c in ("period", "model", "horizon_min", "fresh_version", "fresh_skill", "fresh_skill_twin",
                            "fresh_difference", "slope_per_week", "slope_per_week_twin", "useful_life_days",
                            "useful_life_days_twin") if c in twins.columns]
        lines += ["## Insulin and carbohydrates (D90): each model against its glucose-only twin (skill; slope of the "
                  "frozen model per week)", "", markdown_table(twins[show].round(4)), ""]
    lines += ["## Checks to look at", "",
              "- `ageing_fits.png`: whether a straight line describes the weekly means (curvature would call for a "
              "spline in the paper).",
              "- `ageing_slopes.png` and `version_effects_mixed.png`: which effects exceed the DMD of their model.",
              "- `screening.png`: the models that go on to stage 2.", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
