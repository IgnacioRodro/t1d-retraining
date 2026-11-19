"""31 - Stage 2 analysis: growth of the gain of refitting, optimal interval, later anchors, confirmation policies.

Reads the day metrics and the fit log of script 30 (main anchor and the later
anchors of D83) and answers Q2 (HANDOFF 7.4; ``eval.policy_analysis``):

- **policy table**: per model and policy on every patient of the main anchor,
  mean skill per horizon, refits per week, computing seconds per test day and
  the front of skill against cost;
- **optimal interval** (D84): the growing window every k days (1, 3, 7, 14)
  against every day; the longest k that beats never and stays within the
  model's DMD of every day at each horizon, and the shortest of those per
  model; with the share of the daily gain it keeps and the loss in mg/dL;
- **growth of the gain**: the daily gain of each policy over never, week by
  week since the anchor, and a mixed model of how fast it grows;
- **later anchors** (D83): the gain over never at anchors 14, 42 and 70 (the
  first fit uses every day since day 0) on the patients with every anchor;
  the change of the gain from anchor 14; and the data hypothesis, a mixed
  model of the daily gain on log2 of how much the data have grown since the
  first fit, with and without the anchor;
- **insulin and carbohydrates** (D90): whether refitting gains more when
  the model reads them, from the glucose-only twins of script 30, which run the
  policies of their full models; every table above has the twins too (the
  figures show the full models);
- **confirmation policies** (step 3 of 7.4, D84): per model, never, the
  growing window at the optimal interval and every day (ARX: also online),
  written to ``confirmation.yaml`` for the confirmation scripts (50), with the
  model's twin, which runs the same policies (D90).

Every evaluation also reports RMSE and the Clarke zones (rule R6) per model,
policy, anchor and horizon; the Clarke grids are in script 30. The mixed
models run in parallel on all cores but two (rule R5).

Inputs:
    <derived_data>/runs/30_stage2_policy_runs/day_metrics.parquet and fit_log.parquet
    results/12_stage0_population_transfer/noise_floor.csv   (DMD, D75)

Outputs (results/31_stage2_policy_analysis/):
    policy_table.csv         per model and policy (main anchor): skill, refits per week, seconds per day, front
    interval_comparison.csv  every growing interval against every day and against never
    optimal_interval.csv     per model and horizon: the optimal interval, its loss and the share of the gain it keeps
    weekly_gain.csv          daily gain over never by week since the anchor (patient means, 95 % interval)
    gain_growth.csv          mixed model of the daily gain over the weeks since the anchor
    later_anchor_gains.csv   D83: gain over never per anchor on the patients with every anchor
    anchor_change.csv        D83: change of the gain from the main anchor to each later one
    data_ratio_fits.csv      D83: daily gain ~ log2(data ratio) (and + anchor), per model and horizon
    data_ratio_curve.csv     binned daily gain against log2(data ratio), per model and anchor
    exog_refit.csv           D90: gain over never with every input and with the glucose only, and the difference
    metrics_by_policy.csv    per model, policy, anchor and horizon: n, RMSE, MAE, MARD, bias, Clarke zones (R6)
    fit_log.csv              the mixed models: specification, notes, seconds
    confirmation.yaml        D84: the policies of each model for the confirmation (also in <derived_data>/runs/31_...)
    weekly_gain.png, optimal_interval.png, later_anchors.png, data_ratio.png, pareto.png, exog_refit.png
    summary.md, run_manifest.json

Usage:
    python scripts/31_stage2_policy_analysis.py
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
from t1dretrain.eval import policy_analysis as pa  # noqa: E402
from t1dretrain.eval import policy_report as pr  # noqa: E402
from t1dretrain.eval.stage0_report import (PERIOD_LABELS, SUMS, DMDTable, color, finish, label,  # noqa: E402
                                           read_noise_floor)
from t1dretrain.outputs import ScriptOutput, markdown_table  # noqa: E402
from t1dretrain.plotting import SEQUENTIAL_BLUE, TEXT_SECONDARY, model_axes, set_style  # noqa: E402
from t1dretrain.sim.policies import online_policies, parse_policy  # noqa: E402
from t1dretrain.sim.stage0 import FAMILIES, split_name, split_variant  # noqa: E402

SCRIPT_ID = "31_stage2_policy_analysis"
SOURCE = "30_stage2_policy_runs"
HORIZON_COLORS = {30: SEQUENTIAL_BLUE[2], 60: SEQUENTIAL_BLUE[4], 120: SEQUENTIAL_BLUE[6]}
ANCHOR_COLORS = [SEQUENTIAL_BLUE[1], SEQUENTIAL_BLUE[3], SEQUENTIAL_BLUE[5], SEQUENTIAL_BLUE[6]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--paths", type=Path, default=CONFIG_DIR / "paths.yaml", help="paths file (config/paths.yaml)")
    parser.add_argument("--skip-code-check", action="store_true",
                        help="run even if the code differs from the released manifest")
    return parser.parse_args()


# ============================================================================ jobs (module level for the workers)


def run_fits(job: dict) -> dict:
    """The mixed models of one model and horizon: growth of the gain per policy and anchor, and the data ratio."""
    import time

    started = time.time()
    with WarningRecorder() as recorder:
        try:
            growth = pa.growth_table(job["gain"]).to_dict("records") if not job["gain"].empty else []
            ratio = pa.data_ratio_fits(job["ratio_gain"], job["history"]).to_dict("records") \
                if not job["ratio_gain"].empty else []
            error = ""
        except Exception as exc:  # noqa: BLE001
            growth, ratio, error = [], [], f"{type(exc).__name__}: {exc}"[:300]
    return {"model": job["model"], "horizon_min": job["horizon_min"], "growth": growth, "ratio": ratio,
            "warnings": recorder.text()[:300], "error": error, "seconds": round(time.time() - started, 1)}


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
    stage2, cfg = config["stage2"], config.get("stage2_analysis", {})
    replicates, seed = int(config["stage0"]["bootstrap_replicates"]), int(config["random_seed"])
    min_origins, clip = int(cfg.get("min_origins_per_day", 12)), tuple(cfg.get("skill_clip", [-1.0, 1.0]))
    intervals = [int(k) for k in cfg.get("intervals", [1, 3, 7, 14])]
    window_days, test_days = int(config["anchors"]["train_window_days"]), int(config["anchors"]["max_age_days"])
    main_days = [int(a) for a in stage2["anchor_days"]]
    base_anchor = main_days[0]
    later_cfg = stage2.get("later_anchors") or {}
    period = list(stage2["periods"])[0]

    runs = derived / "runs" / SOURCE
    if not (runs / "day_metrics.parquet").exists():
        raise SystemExit(f"{runs / 'day_metrics.parquet'} not found: run scripts/30_stage2_policy_runs.py first")
    days = pd.read_parquet(runs / "day_metrics.parquet")
    days = days[days["period"] == period]
    fit_log = pd.read_parquet(runs / "fit_log.parquet") if (runs / "fit_log.parquet").exists() else pd.DataFrame()
    every_model = run_models(out, days)
    models = [m for m in every_model if not split_variant(m)[1]]  # the full models; their glucose-only twins (D90)
    twins = {split_variant(m)[0]: m for m in every_model if split_variant(m)[1]}  # get every table, not the figures
    dmd = DMDTable(read_noise_floor(Path(out.dir).parent / "12_stage0_population_transfer" / "noise_floor.csv"),
                   [m for m in every_model if split_name(m)[0] in FAMILIES])
    online = {m: online_policies(m) for m in models}
    print(f"{len(days)} day rows; models {models}; anchors {sorted(int(a) for a in days['anchor_day'].unique())}")

    # ---- main anchor: policy table, optimal interval, growth of the gain
    main = days[days["anchor_day"].isin(main_days)]
    skill = complete(pr.patient_skill(main))
    used = fit_log[fit_log["anchor_day"].isin(main_days)] if not fit_log.empty else fit_log
    if "excluded" in used:
        used = used[used["excluded"].isna()]
    kinds = {p: parse_policy(p).kind for p in skill["policy"].unique()}
    table = pa.policy_table(skill, used, test_days, kinds)
    comparison = pa.interval_table(skill, intervals, dmd, replicates, seed)
    optimal = pa.optimal_intervals(comparison) if not comparison.empty else pd.DataFrame()
    curve_policies = [pa.interval_policy(k) for k in intervals] + ["online"]
    gain = pa.daily_gain(main, curve_policies, "never", min_origins, clip)
    weekly = pa.weekly_gain(gain) if not gain.empty else pd.DataFrame()

    # ---- later anchors (D83)
    later_days = sorted(set(main_days) | {int(a) for a in later_cfg.get("anchor_days", [])})
    later_days = [a for a in later_days if a in set(days["anchor_day"].unique())]
    history = {a: (window_days if a in main_days else a) for a in later_days}
    later_gains, change, ratio_gain, curve = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    n_common = {}
    if len(later_days) > 1:
        lat = days[days["anchor_day"].isin(later_days)]
        lat_skill = pr.patient_skill(lat, by=("anchor_day",))
        daily = pa.interval_policy(1)
        common = pa.common_patients(lat_skill, later_days, ["never", daily])
        n_common = {m: len(common.get((period, m), [])) for m in every_model}
        keep = lat_skill.apply(lambda r: r["patient_id"] in set(common.get((r["period"], r["model"]), [])), axis=1)
        lat_skill = lat_skill[keep]
        later_gains = pr.paired(lat_skill, lambda p: "never", dmd, replicates, seed, by=("anchor_day",))
        change = pd.concat([pa.anchor_change(lat_skill, p, base_anchor, dmd, replicates, seed)
                            for p in sorted(set(lat_skill["policy"]) - {"never"})], ignore_index=True)
        lat_gain = pa.daily_gain(lat, [daily], "never", min_origins, clip)
        pairs = {(m, p) for (_, m), ps in common.items() for p in ps}
        ratio_gain = lat_gain[[(m, p) in pairs for m, p in zip(lat_gain["model"], lat_gain["patient_id"])]]
        curve = pa.ratio_curve(ratio_gain, history) if not ratio_gain.empty else pd.DataFrame()

    # ---- mixed models in parallel
    growth_policies = [pa.interval_policy(1), pa.interval_policy(max(intervals)), "online"]
    jobs = []
    for model in every_model:
        for horizon in sorted(days["horizon_min"].unique()):
            g = gain[(gain["model"] == model) & (gain["horizon_min"] == horizon) & gain["policy"].isin(growth_policies)]
            r = ratio_gain[(ratio_gain["model"] == model) & (ratio_gain["horizon_min"] == horizon)] \
                if not ratio_gain.empty else pd.DataFrame()
            jobs.append({"model": model, "horizon_min": int(horizon), "gain": g, "ratio_gain": r, "history": history})
    results = parallel_map(run_fits, jobs, resources.n_workers, label="mixed models")
    growth = pd.DataFrame([row for r in results for row in r["growth"]])
    ratio = pd.DataFrame([row for r in results for row in r["ratio"]])
    fit_rows = pd.DataFrame([{k: r[k] for k in ("model", "horizon_min", "warnings", "error", "seconds")}
                             for r in results])
    if not growth.empty:
        fit_rows = fit_rows.merge(growth.groupby(["model", "horizon_min"]).agg(
            growth_specs=("spec", lambda s: "; ".join(sorted(set(map(str, s))))),
            growth_notes=("note", lambda s: " | ".join(sorted({str(x) for x in s if isinstance(x, str) and x})))
        ).reset_index(), on=["model", "horizon_min"], how="left")
    if not ratio.empty and "spec" in ratio:
        fit_rows = fit_rows.merge(ratio[["model", "horizon_min", "spec"]].rename(columns={"spec": "ratio_spec"}),
                                  on=["model", "horizon_min"], how="left")

    # ---- insulin and carbohydrates (D90)
    exog = pa.exog_effect(skill, dmd, replicates, seed)
    exog_gains = pr.paired(skill, lambda p: "never", dmd, replicates, seed) if twins else pd.DataFrame()

    # ---- confirmation (D84) and R6
    confirmation = pa.confirmation_policies(optimal, models, online) if not optimal.empty else {}
    for model, entry in confirmation.items():  # D90: the twin runs the policies of its full model
        if model in twins:
            entry["twin"] = twins[model]
    document = {"release": release, "period": period, "anchor_days": main_days,
                "rule": "never, the growing window at the model's optimal interval (the longest that beats never and is "
                        "within the DMD of every day; the shortest over horizons) and every day; ARX also online (D84)",
                "models": confirmation}
    (derived / "runs" / SCRIPT_ID).mkdir(parents=True, exist_ok=True)
    (derived / "runs" / SCRIPT_ID / "confirmation.yaml").write_text(yaml.safe_dump(document, sort_keys=False))
    metrics = finish(days.groupby(["period", "model", "policy", "anchor_day", "horizon_min"], as_index=False)[SUMS]
                     .sum()).round(4)

    out.csv(table.round(4), "policy_table")
    out.csv(comparison.round(4), "interval_comparison")
    out.csv(optimal.round(4), "optimal_interval")
    out.csv(weekly.round(4), "weekly_gain")
    out.csv(growth.round(5), "gain_growth")
    out.csv(later_gains.round(4), "later_anchor_gains")
    out.csv(change.round(4), "anchor_change")
    out.csv(ratio.round(5), "data_ratio_fits")
    out.csv(curve.round(4), "data_ratio_curve")
    out.csv(metrics, "metrics_by_policy")
    out.csv(fit_rows, "fit_log")
    out.csv(exog.round(4), "exog_refit")
    out.text(yaml.safe_dump(document, sort_keys=False), "confirmation.yaml")
    if not weekly.empty:
        out.figure(plot_weekly(weekly, models, intervals, base_anchor, period), "weekly_gain")
    if not comparison.empty:
        out.figure(plot_intervals(comparison, optimal, models, period), "optimal_interval")
    if not later_gains.empty:
        out.figure(plot_later(later_gains, models, n_common, period), "later_anchors")
    if not curve.empty:
        out.figure(plot_ratio(curve, ratio, models, period), "data_ratio")
    if not table.empty:
        out.figure(plot_front(table, models, period), "pareto")
    if not exog.empty:
        out.figure(plot_exog(exog_gains, exog, period), "exog_refit")
    out.text(write_summary(table, comparison, optimal, growth, later_gains, change, ratio, exog, fit_rows, document,
                           models, later_days, n_common, history, release, resources, period, len(days)), "summary.md")
    out.manifest(settings={"period": period, "main_anchor_days": main_days, "later_anchor_days": later_days,
                           "history_days": history, "intervals": intervals, "min_origins_per_day": min_origins,
                           "skill_clip": list(clip), "models": models, "bootstrap_replicates": replicates},
                 notes="Inputs: day metrics and fit log of script 30; DMD from script 12.")
    print(f"Done. Results in {out.dir}")


def run_models(out: ScriptOutput, days: pd.DataFrame) -> list[str]:
    """The models of script 30, in its order (its run manifest), else as they appear."""
    import json

    path = Path(out.dir).parent / SOURCE / "run_manifest.json"
    present = [m for m in days["model"].unique() if m != "persistence"]
    try:
        order = json.loads(path.read_text())["settings"]["models"]
    except Exception:  # noqa: BLE001
        order = []
    return [m for m in order if m in present] + [m for m in present if m not in order]


def complete(skill: pd.DataFrame) -> pd.DataFrame:
    """Only the policies that ran on every patient of their model (round 1 alone ran on 10)."""
    if skill.empty:
        return skill
    patients = skill.groupby(["period", "model", "policy"])["patient_id"].nunique().rename("n").reset_index()
    full = patients.groupby(["period", "model"])["n"].transform("max")
    keep = patients[patients["n"] == full][["period", "model", "policy"]]
    return skill.merge(keep, on=["period", "model", "policy"])


# ============================================================================ figures


def plot_weekly(weekly: pd.DataFrame, models: list[str], intervals: list[int], anchor: int, period: str):
    w = weekly[(weekly["period"] == period) & (weekly["anchor_day"] == anchor)]
    fig, axes = model_axes(len(models), 4.4, 3.4)
    styles = {pa.interval_policy(1): "-", pa.interval_policy(max(intervals)): "--", "online": ":"}
    for ax, model in zip(axes, models):
        m = w[w["model"] == model]
        for horizon in sorted(m["horizon_min"].unique()):
            for policy, style in styles.items():
                c = m[(m["horizon_min"] == horizon) & (m["policy"] == policy)].sort_values("week")
                if c.empty:
                    continue
                ax.plot(c["week"], 100 * c["gain_mean"], ls=style, color=HORIZON_COLORS.get(horizon, TEXT_SECONDARY),
                        lw=1.6, label=f"{horizon} min, {policy}")
                if style == "-":
                    ax.fill_between(c["week"], 100 * c["gain_low"], 100 * c["gain_high"], alpha=0.12, lw=0,
                                    color=HORIZON_COLORS.get(horizon, TEXT_SECONDARY))
        ax.axhline(0, color=TEXT_SECONDARY, lw=0.8)
        ax.set_title(label(model), loc="left", fontsize=9)
        ax.set_xlabel("weeks since the anchor")
    axes[0].set_ylabel("daily gain over never (points)")
    axes[0].legend(fontsize=6, ncol=2, loc="upper left")
    fig.suptitle(f"How the gain of refitting grows with the weeks (growing window; solid: every day, dashed: every "
                 f"{max(intervals)} days, dotted: online), {PERIOD_LABELS.get(period, period)}", fontweight="bold",
                 x=0.01, ha="left")
    fig.tight_layout()
    return fig


def plot_intervals(comparison: pd.DataFrame, optimal: pd.DataFrame, models: list[str], period: str):
    c = comparison[comparison["period"] == period]
    fig, axes = model_axes(len(models), 4.2, 3.4)
    for ax, model in zip(axes, models):
        m = c[c["model"] == model]
        for horizon in sorted(m["horizon_min"].unique()):
            h = m[m["horizon_min"] == horizon].sort_values("interval_days")
            colour = HORIZON_COLORS.get(horizon, TEXT_SECONDARY)
            y = 100 * h["difference_to_daily"].fillna(0.0)
            ax.plot(h["interval_days"], y, color=colour, lw=1.4, label=f"{horizon} min")
            low = 100 * h["daily_low"].fillna(h["difference_to_daily"]) if "daily_low" in h else y
            high = 100 * h["daily_high"].fillna(h["difference_to_daily"]) if "daily_high" in h else y
            ax.vlines(h["interval_days"], low, high, color=colour, lw=1.2)
            tie = h["verdict_to_daily"].isin(["tie", "reference"])
            ax.scatter(h["interval_days"][tie], y[tie], s=26, facecolors="white", edgecolors=colour, zorder=3)
            ax.scatter(h["interval_days"][~tie], y[~tie], s=26, color=colour, zorder=3)
        ax.axhline(0, color=TEXT_SECONDARY, lw=0.8)
        ax.set_xscale("log", base=2)
        ticks = sorted(m["interval_days"].dropna().unique())
        ax.set_xticks(ticks, [f"{int(k)} d" for k in ticks])
        o = optimal[(optimal["period"] == period) & (optimal["model"] == model)]
        k = o["model_interval_days"].iloc[0] if not o.empty else np.nan
        ax.set_title(f"{label(model)}: " + (f"every {int(k)} d" if pd.notna(k) else "never is enough"), loc="left",
                     fontsize=9)
        ax.set_xlabel("refit every (growing window)")
    axes[0].set_ylabel("skill points against every day")
    axes[0].legend(fontsize=7, loc="lower left")
    fig.suptitle(f"Optimal interval: the longest that beats never and stays within the DMD of refitting every day "
                 f"(hollow: within), {PERIOD_LABELS.get(period, period)}", fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    return fig


def plot_later(gains: pd.DataFrame, models: list[str], n_common: dict, period: str, policy: str = "growing_every1"):
    g = gains[(gains["period"] == period) & (gains["policy"] == policy)]
    fig, axes = model_axes(len(models), 4.2, 3.4)
    for ax, model in zip(axes, models):
        m = g[g["model"] == model]
        for horizon in sorted(m["horizon_min"].unique()):
            h = m[m["horizon_min"] == horizon].sort_values("anchor_day")
            colour = HORIZON_COLORS.get(horizon, TEXT_SECONDARY)
            ax.errorbar(h["anchor_day"], 100 * h["skill_difference"],
                        yerr=[100 * (h["skill_difference"] - h["low"]), 100 * (h["high"] - h["skill_difference"])],
                        color=colour, marker="o", ms=4, lw=1.4, capsize=2, label=f"{horizon} min")
        ax.axhline(0, color=TEXT_SECONDARY, lw=0.8)
        ax.set_xticks(sorted(g["anchor_day"].unique()))
        ax.set_title(f"{label(model)} ({n_common.get(model, 0)} patients)", loc="left", fontsize=9)
        ax.set_xlabel("anchor day = days of history at the first fit")
    axes[0].set_ylabel("skill points over never (95 % CI)")
    axes[0].legend(fontsize=7)
    fig.suptitle(f"Gain of refitting every day by the history at the start (D83; patients with every anchor), "
                 f"{PERIOD_LABELS.get(period, period)}", fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    return fig


def plot_ratio(curve: pd.DataFrame, ratio: pd.DataFrame, models: list[str], period: str):
    c = curve[curve["period"] == period]
    anchors = sorted(c["anchor_day"].unique())
    fig, axes = model_axes(len(models), 4.2, 3.4)
    for ax, model in zip(axes, models):
        m = c[c["model"] == model]
        for k, anchor in enumerate(anchors):
            a = m[m["anchor_day"] == anchor].sort_values("log2_ratio")
            ax.plot(a["log2_ratio"], 100 * a["gain_mean"], marker="o", ms=3, lw=1.4,
                    color=ANCHOR_COLORS[min(k, len(ANCHOR_COLORS) - 1)], label=f"anchor {anchor}")
        r = ratio[(ratio["model"] == model) & (ratio["horizon_min"] == 60)] if not ratio.empty else ratio
        slope = f"; 60 min: {100 * r['per_doubling'].iloc[0]:+.2f} per doubling" \
            if not r.empty and "per_doubling" in r and np.isfinite(r["per_doubling"].iloc[0]) else ""
        ax.axhline(0, color=TEXT_SECONDARY, lw=0.8)
        ax.set_title(f"{label(model)}{slope}", loc="left", fontsize=9)
        ax.set_xlabel("log2(data at the refit / data at the first fit)")
    axes[0].set_ylabel("daily gain over never, mean of horizons (points)")
    axes[0].legend(fontsize=7)
    fig.suptitle(f"Data hypothesis (D83): the gain against how much the data have grown, by anchor, "
                 f"{PERIOD_LABELS.get(period, period)}", fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    return fig


def plot_front(table: pd.DataFrame, models: list[str], period: str):
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    t = table[(table["period"] == period) & (table["seconds_per_day"] > 0)]
    fig, axes = model_axes(len(models), 4.4, 3.6)
    for ax, model in zip(axes, models):
        m = t[t["model"] == model]
        fixed = m[m["kind"].isin(["never", "fixed"])]
        ax.scatter(fixed["seconds_per_day"], 100 * fixed["skill_mean"], color=color(model), s=16, alpha=0.6)
        for kind, marker in (("trigger", "^"), ("online", "D")):
            k = m[m["kind"] == kind]
            if not k.empty:
                ax.scatter(k["seconds_per_day"], 100 * k["skill_mean"], facecolors="none", edgecolors=color(model),
                           marker=marker, s=28, label=kind)
        front = fixed[fixed["front"]].sort_values("seconds_per_day")
        ax.plot(front["seconds_per_day"], 100 * front["skill_mean"], color=color(model), lw=1.4)
        for _, r in front.iterrows():
            ax.annotate(r["policy"].replace("growing_", ""), (r["seconds_per_day"], 100 * r["skill_mean"]),
                        fontsize=6, textcoords="offset points", xytext=(3, -8))
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(LogLocator(base=10))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _p: f"{x:g} s"))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_title(label(model), loc="left", fontsize=9)
        ax.set_xlabel("computing time per test day")
        if len(m[m["kind"].isin(["trigger", "online"])]):
            ax.legend(fontsize=6, loc="lower right")
    axes[0].set_ylabel("mean skill over horizons (%)")
    fig.suptitle(f"Skill against cost on every patient (front of the fixed policies), "
                 f"{PERIOD_LABELS.get(period, period)}", fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    return fig


def plot_exog(gains: pd.DataFrame, exog: pd.DataFrame, period: str, policy: str = "growing_every1"):
    """D90: gain of refitting every day over never, with every input and with the glucose only."""
    pairs = sorted(set(zip(exog["model"], exog["twin"])))
    fig, axes = model_axes(len(pairs), 4.0, 3.4)
    g = gains[(gains["period"] == period) & (gains["policy"] == policy)]
    for ax, (base, twin) in zip(axes, pairs):
        for k, (model, name) in enumerate(((base, "every input"), (twin, "glucose only"))):
            m = g[g["model"] == model].sort_values("horizon_min")
            x = np.arange(len(m)) + (k - 0.5) * 0.18
            ax.errorbar(x, 100 * m["skill_difference"], yerr=[100 * (m["skill_difference"] - m["low"]),
                        100 * (m["high"] - m["skill_difference"])], fmt="o" if k == 0 else "D", ms=5, capsize=2,
                        lw=1.4, color=color(base), mfc=color(base) if k == 0 else "white", label=name)
            ax.set_xticks(np.arange(len(m)), [f"{int(h)} min" for h in m["horizon_min"]])
        ax.axhline(0, color=TEXT_SECONDARY, lw=0.8)
        ax.set_title(label(base), loc="left", fontsize=9)
    axes[0].set_ylabel("skill points over never (95 % CI)")
    axes[0].legend(fontsize=7)
    fig.suptitle(f"Refitting every day with and without insulin and carbohydrates (D90; filled: every input, "
                 f"hollow: glucose only), "
                 f"{PERIOD_LABELS.get(period, period)}", fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    return fig


# ============================================================================ summary


def write_summary(table, comparison, optimal, growth, later, change, ratio, exog, fit_rows, document, models,
                  later_days, n_common, history, release, resources, period, n_rows) -> str:
    lines = ["# 31 - Stage 2 analysis: how often and with which window", "", f"Code release {release}.",
             f"Machine: {resources.describe()}",
             f"{n_rows} day rows of script 30; models {models}; {PERIOD_LABELS.get(period, period)}; anchors "
             f"{later_days} (days of history at the first fit: {history}).", ""]
    problems = [f"{r.model} {r.horizon_min} min failed: {r.error}" for r in fit_rows.itertuples() if r.error]
    problems += [f"{r.model} {r.horizon_min} min raised warnings: {r.warnings}" for r in fit_rows.itertuples()
                 if isinstance(r.warnings, str) and r.warnings]
    if not growth.empty and "note" in growth:
        problems += [f"growth {r.model} {r.policy} {r.horizon_min} min (anchor {r.anchor_day}): {r.note}"
                     for r in growth.itertuples() if isinstance(r.note, str) and "not converged" in r.note][:12]
    lines += ["## Warnings", ""] + ([f"- {p}" for p in problems] or ["- none"]) + [""]
    if not optimal.empty:
        o = optimal[optimal["period"] == period].copy()
        for c in ("difference_to_daily", "gain_over_never", "share_of_daily_gain"):
            if c in o:
                o[c] = (100 * o[c]).round(2)
        lines += ["## Optimal interval (D84)", "",
                  "The longest refit interval of the growing window that beats never and whose loss against every "
                  "day is within the model's DMD; per model, the shortest over horizons (`model_interval_days`; "
                  "empty: never is enough). Points of skill; `share_of_daily_gain` in %. A glucose-only twin's own "
                  "interval is shown for reference: the twin runs its full model's (D90).", "",
                  markdown_table(o[[c for c in ("model", "horizon_min", "optimal_policy", "optimal_interval_days",
                                                "difference_to_daily", "mgdl_to_daily", "gain_over_never",
                                                "share_of_daily_gain", "model_interval_days") if c in o]].round(3)),
                  "",
                  "Every interval against every day and against never is in `interval_comparison.csv`.", ""]
    if not growth.empty and "growth_per_week" in growth:
        g = growth[(growth["period"] == period) & (growth["policy"] == "growing_every1")].copy()
        for c in ("gain_at_anchor", "growth_per_week", "growth_low", "growth_high"):
            g[c] = (100 * g[c]).round(3)
        lines += ["## Growth of the gain of refitting every day (points; per week since the anchor)", "",
                  markdown_table(g[["model", "horizon_min", "anchor_day", "gain_at_anchor", "growth_per_week",
                                    "growth_low", "growth_high", "spec"]]), ""]
    if not later.empty:
        t = later[(later["period"] == period) & (later["policy"] == "growing_every1")].copy()
        t["cell"] = [f"{100 * d:+.2f} ({lo * 100:+.2f} to {hi * 100:+.2f}; {v})" for d, lo, hi, v in
                     zip(t["skill_difference"], t["low"], t["high"], t["verdict"])]
        wide = t.pivot_table(index=["model", "horizon_min"], columns="anchor_day", values="cell",
                             aggfunc="first").reset_index()
        wide.columns = [f"anchor {c}" if isinstance(c, (int, np.integer)) else c for c in wide.columns]
        wide.insert(1, "patients", wide["model"].map(n_common))
        lines += ["## Later anchors (D83): gain of refitting every day over never", "",
                  "Patients with every anchor; at each anchor the first fit uses every day since day 0.", "",
                  markdown_table(wide), ""]
    if not change.empty:
        c = change[(change["period"] == period) & (change["policy"] == "growing_every1")].copy()
        for col in ("gain_base", "gain_anchor", "change", "low", "high", "dmd"):
            c[col] = (100 * c[col]).round(2)
        lines += ["### Change of the gain from the main anchor (points; negative: smaller with more history)", "",
                  markdown_table(c[["model", "horizon_min", "anchor_day", "patients", "gain_base", "gain_anchor",
                                    "change", "low", "high", "dmd", "verdict"]]), ""]
    if not ratio.empty and "per_doubling" in ratio:
        r = ratio[ratio["period"] == period].copy()
        anchor_cols = [c for c in r.columns if c.startswith("anchor_") and not c.endswith(("_low", "_high"))]
        show = ["model", "horizon_min", "per_doubling", "per_doubling_low", "per_doubling_high",
                "per_doubling_with_anchor"] + anchor_cols + ["spec"]
        for col in show[2:-1]:
            r[col] = (100 * r[col]).round(3)
        lines += ["### Data hypothesis: daily gain ~ log2(data ratio) (points per doubling of the data; anchor terms "
                  "against the main anchor in the second model)", "", markdown_table(r[[c for c in show if c in r]]),
                  ""]
    if not exog.empty:
        e = exog[(exog["period"] == period) & exog["policy"].isin(["growing_every1", "growing_every14", "online"])]
        e = e.copy()
        for col in ("gain_all_inputs", "gain_glucose_only", "difference", "low", "high", "dmd"):
            e[col] = (100 * e[col]).round(2)
        lines += ["## Does refitting gain more with insulin and carbohydrates? (D90)", "",
                  "Gain over never of the full model and of its glucose-only twin (points, same patients) and "
                  "their paired difference (positive: refitting gains more with insulin and carbohydrates). Every "
                  "policy is in `exog_refit.csv`.", "",
                  markdown_table(e[["model", "policy", "horizon_min", "patients", "gain_all_inputs",
                                    "gain_glucose_only", "difference", "low", "high", "dmd", "verdict"]]), ""]
    if document.get("models"):
        rows = [{"model": m, "optimal_interval_days": v["optimal_interval_days"], "policies": ", ".join(v["policies"])}
                for m, v in document["models"].items()]
        lines += ["## Policies for the confirmation (D84; `confirmation.yaml`)", "", markdown_table(pd.DataFrame(rows)),
                  ""]
    if not table.empty:
        t = table[table["period"] == period]
        show = [c for c in ("model", "policy", "kind", "skill_mean", "skill_30", "skill_60", "skill_120",
                            "refits_per_week", "seconds_per_day", "front", "patients") if c in t.columns]
        lines += ["## Policies on every patient of the main anchor", "", markdown_table(t[show].round(4)), ""]
    lines += ["## Checks to look at", "",
              "- `weekly_gain.png`: whether the gain keeps growing over the 12 weeks or levels off.",
              "- `optimal_interval.png`: how much each longer interval loses against every day.",
              "- `later_anchors.png` and `data_ratio.png`: whether the gain shrinks with more history at the start, "
              "and whether the curves of the anchors fall on one line against the data ratio.", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
