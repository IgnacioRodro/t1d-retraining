"""51 - Confirmation analysis: does what the screening found hold on new patients and on other datasets?

Reads the day metrics and the fit log of script 50 and, per cohort, sets each
finding of the screening against its confirmation (HANDOFF 7.6, D92;
``eval.confirmation``). The comparisons were fixed before the confirmation
was run:

- **refitting pays** (stage 2, D84): the growing window at the model's optimal
  interval (``restart_every<k>``) against never;
- **the optimal interval is enough** (D84): ``restart_every<k>`` against every
  day (``restart_every1``, where the cohort runs it); replicated only when the
  loss is within the DMD again (tie, or better);
- **the mechanism** (stage 3, D91): the mechanism of script 41 at the interval
  against the restart (ARX: its update at every sample against the batch refit,
  reported without a screening value, as stage 2 did not pair them);
- a finding that was better in the screening is replicated when it is better
  again (beyond the model's DMD, with an interval that excludes zero), and one
  that tied when it is not worse;
- **insulin and carbohydrates** (D90): the gain of every arm over never with
  every input and with the glucose only (the twin), paired per patient;
- **over the weeks**: the daily gain over never by week since the anchor, and
  how much it grows per week (mixed model, as in script 31), up to 36 weeks in
  Loop;
- **forecast lows**: sensitivity and precision below 70 and 54 mg/dL of every
  arm (HANDOFF 7.8), pooled over forecasts.

The screening values come from scripts 31 (``interval_comparison.csv``) and 41
(``vs_reference.csv``) on the screen group of the core dataset. Every
evaluation reports RMSE and the Clarke zones (rule R6); the grid data are in
script 50.

Inputs:
    <derived_data>/runs/50_confirmation_runs/day_metrics.parquet and fit_log.parquet
    results/50_confirmation_runs/run_manifest.json        (plan, arms and cohorts)
    results/31_stage2_policy_analysis/interval_comparison.csv, results/41_stage3_mechanism_analysis/vs_reference.csv
    results/12_stage0_population_transfer/noise_floor.csv  (DMD, D75)

Outputs (results/51_confirmation_analysis/):
    replication.csv          per cohort, model, horizon and finding: screening and confirmation, verdicts, replicated
    replication_summary.csv  per cohort and finding: comparisons replicated / made
    exog_confirmation.csv    D90: gain over never of every arm with every input and with the glucose only
    weekly_gain.csv          daily gain over never by week since the anchor (patient means, 95 %)
    gain_growth.csv          growth of the gain per week (mixed model)
    low_glucose.csv          sensitivity and precision of the forecast lows per cohort, model, arm and horizon
    metrics_by_arm.csv       per cohort, model, arm and horizon: n, RMSE, MAE, MARD, bias, Clarke zones (R6)
    fit_log.csv              the mixed models of the growth (specification, warnings)
    replication.png, weekly_gain.png, exog_confirmation.png, low_glucose.png
    summary.md, run_manifest.json

Usage:
    python scripts/51_confirmation_analysis.py
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

from t1dretrain.compute import detect_resources, parallel_map  # noqa: E402
from t1dretrain.config import CONFIG_DIR, derived_data_dir, load_project_config  # noqa: E402
from t1dretrain.data.pipeline import WarningRecorder  # noqa: E402
from t1dretrain.eval import confirmation as cf  # noqa: E402
from t1dretrain.eval import policy_analysis as pa  # noqa: E402
from t1dretrain.eval import policy_report as pr  # noqa: E402
from t1dretrain.eval.stage0_report import SUMS, DMDTable, color, finish, label, read_noise_floor  # noqa: E402
from t1dretrain.outputs import ScriptOutput, markdown_table  # noqa: E402
from t1dretrain.plotting import TEXT_SECONDARY, model_axes, set_style  # noqa: E402
from t1dretrain.sim.stage0 import FAMILIES, is_twin, split_name, twin_of  # noqa: E402

SCRIPT_ID = "51_confirmation_analysis"
SOURCE = "50_confirmation_runs"
FINDINGS = {"gain_over_never": "refitting at the optimal interval beats never",
            "loss_to_daily": "the optimal interval loses no more than the DMD to every day",
            "mechanism_vs_restart": "the mechanism of script 41 against the restart"}
SCREEN = "screen (core)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--paths", type=Path, default=CONFIG_DIR / "paths.yaml", help="paths file (config/paths.yaml)")
    parser.add_argument("--skip-code-check", action="store_true",
                        help="run even if the code differs from the released manifest")
    return parser.parse_args()


def run_growth(job: dict) -> dict:
    """The growth of the gain over never (mixed models) of one cohort and model."""
    import time

    started = time.time()
    with WarningRecorder() as recorder:
        try:
            rows = pa.growth_table(job["gain"]).to_dict("records") if not job["gain"].empty else []
            error = ""
        except Exception as exc:  # noqa: BLE001
            rows, error = [], f"{type(exc).__name__}: {exc}"[:300]
    return {"cohort": job["cohort"], "model": job["model"], "rows": rows, "warnings": recorder.text()[:300],
            "error": error, "seconds": round(time.time() - started, 1)}


# ============================================================================ main


def main() -> None:
    import json

    args = parse_args()
    release = verify_code(strict=not args.skip_code_check)
    config = load_project_config()
    resources = detect_resources(config.get("compute", {}).get("reserve_cores"))
    print(resources.describe())
    out = ScriptOutput(SCRIPT_ID, resources=resources)
    set_style()
    derived = derived_data_dir(args.paths)
    cfg = config.get("stage2_analysis", {})
    replicates, seed = int(config["stage0"]["bootstrap_replicates"]), int(config["random_seed"])
    min_origins, clip = int(cfg.get("min_origins_per_day", 12)), tuple(cfg.get("skill_clip", [-1.0, 1.0]))
    runs = derived / "runs" / SOURCE
    if not (runs / "day_metrics.parquet").exists():
        raise SystemExit(f"{runs / 'day_metrics.parquet'} not found: run scripts/50_confirmation_runs.py first")
    days = pd.read_parquet(runs / "day_metrics.parquet")
    results = Path(out.dir).parent
    manifest = json.loads((results / SOURCE / "run_manifest.json").read_text())
    settings = manifest["settings"]
    plan, arms_of = settings["plan"], settings["arms"]
    cohorts = [cf.cohort_key(c["dataset"], c["cohort"]) for c in settings["cohorts"]]
    cohorts = [c for c in cohorts if c in set(days["cohort"])]
    models = [m for m in settings["models"] if m in set(days["model"])]
    dmd = DMDTable(read_noise_floor(results / "12_stage0_population_transfer" / "noise_floor.csv"),
                   [m for m in models if split_name(m)[0] in FAMILIES])
    screen = screening_values(results, plan)
    print(f"{len(days)} day rows; cohorts {cohorts}; models {models}")

    checks, exog, weekly, lows, metrics, growth_jobs = [], [], [], [], [], []
    sums = SUMS + cf.low_sums(days)
    for cohort in cohorts:
        d = days[days["cohort"] == cohort]
        skill = pr.patient_skill(d)
        checks.append(replication_rows(skill, plan, screen, dmd, replicates, seed).assign(cohort=cohort))
        exog.append(pa.exog_effect(skill, dmd, replicates, seed).assign(cohort=cohort))
        refits = sorted({a for m in arms_of.get(cohort, {}) for a in arms_of[cohort][m] if a != "never"})
        gain = pa.daily_gain(d, refits, "never", min_origins, clip)
        if not gain.empty:
            weekly.append(pa.weekly_gain(gain).assign(cohort=cohort))
            for model, g in gain.groupby("model"):
                k = plan.get(model, {}).get("interval")
                keep = [a for a in g["policy"].unique() if not a.endswith("_every1") or k in (None, 1)]
                growth_jobs.append({"cohort": cohort, "model": model, "gain": g[g["policy"].isin(keep)]})
        pooled = d.groupby(["period", "model", "policy", "horizon_min"], as_index=False)[sums].sum()
        table = cf.with_lows(finish(pooled)).round(4).assign(cohort=cohort)
        metrics.append(table)
        lows.append(low_table(table))
    results_growth = parallel_map(run_growth, growth_jobs, resources.n_workers, label="growth fits") \
        if growth_jobs else []
    growth = first_cohort(pd.DataFrame([dict(r, cohort=x["cohort"]) for x in results_growth for r in x["rows"]]))
    fit_rows = pd.DataFrame([{k: x[k] for k in ("cohort", "model", "warnings", "error", "seconds")}
                             for x in results_growth])
    replication = first_cohort(concat(checks))
    exog = first_cohort(concat(exog))
    weekly = first_cohort(concat(weekly))
    lows = first_cohort(concat(lows))
    metrics = first_cohort(concat(metrics))
    tally = replication_summary(replication)

    out.csv(replication.round(5), "replication")
    out.csv(tally, "replication_summary")
    out.csv(exog.round(4), "exog_confirmation")
    out.csv(weekly.round(4), "weekly_gain")
    out.csv(growth.round(5), "gain_growth")
    out.csv(lows.round(3), "low_glucose")
    out.csv(metrics.drop(columns=[c for c in metrics.columns if c.startswith("low")], errors="ignore"),
            "metrics_by_arm")
    out.csv(fit_rows, "fit_log")
    full = [m for m in models if not is_twin(m)]
    if not replication.empty:
        out.figure(plot_replication(replication, full, cohorts), "replication")
    if not weekly.empty:
        out.figure(plot_weekly(weekly, full, cohorts, plan), "weekly_gain")
    if not exog.empty:
        out.figure(plot_exog(exog, cohorts, plan), "exog_confirmation")
    if not lows.empty:
        out.figure(plot_lows(lows, full, cohorts[0], plan), "low_glucose")
    out.text(write_summary(replication, tally, exog, growth, lows, fit_rows, plan, cohorts, models, manifest, release,
                           resources), "summary.md")
    out.manifest(settings={"cohorts": cohorts, "models": models, "plan": plan, "bootstrap_replicates": replicates,
                           "findings": FINDINGS, "source_release": manifest.get("package_version")},
                 notes="Inputs: day metrics of script 50; screening values from scripts 31 and 41; DMD from 12.")
    print(f"Done. Results in {out.dir}")


def concat(frames: list) -> pd.DataFrame:
    frames = [f for f in frames if f is not None and not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def first_cohort(table: pd.DataFrame) -> pd.DataFrame:
    return table[["cohort"] + [c for c in table.columns if c != "cohort"]] if "cohort" in table else table


# ============================================================================ screening against confirmation


def screening_values(results: Path, plan: dict) -> pd.DataFrame:
    """The screening value of every finding per model and horizon (scripts 31 and 41, core period)."""
    rows = []
    path = results / "31_stage2_policy_analysis" / "interval_comparison.csv"
    if path.exists():
        t = pd.read_csv(path)
        for model, p in plan.items():
            k = p.get("interval")
            if not k:
                continue
            part = t[(t["model"] == model) & (t["policy"] == pa.interval_policy(k))]
            for r in part.itertuples():
                rows.append({"model": model, "horizon_min": int(r.horizon_min), "finding": "gain_over_never",
                             "screen_difference": r.gain_over_never, "screen_verdict": r.verdict_over_never})
                if int(k) != 1:
                    rows.append({"model": model, "horizon_min": int(r.horizon_min), "finding": "loss_to_daily",
                                 "screen_difference": r.difference_to_daily, "screen_verdict": r.verdict_to_daily})
    path = results / "41_stage3_mechanism_analysis" / "vs_reference.csv"
    if path.exists():
        t = pd.read_csv(path)
        for model, p in plan.items():
            mechanism = p.get("mechanism")
            if not mechanism or mechanism == "restart" or mechanism.startswith("online_l"):
                continue
            part = t[(t["model"] == model) & (t["policy"] == mechanism) & (t["against"] == "restart")]
            for r in part.itertuples():
                rows.append({"model": model, "horizon_min": int(r.horizon_min), "finding": "mechanism_vs_restart",
                             "screen_difference": r.skill_difference, "screen_verdict": r.verdict})
    return pd.DataFrame(rows, columns=["model", "horizon_min", "finding", "screen_difference", "screen_verdict"])


def replicated(finding: str, confirm: str, screen: str | None) -> bool | None:
    """D92: the rule of each finding (see the module docstring); None without a screening value."""
    if screen is None:
        return None
    if finding == "loss_to_daily":
        return confirm in ("tie", "better")
    return confirm == "better" if screen == "better" else confirm != "worse"


def replication_rows(skill: pd.DataFrame, plan: dict, screen: pd.DataFrame, dmd, replicates: int,
                     seed: int) -> pd.DataFrame:
    """Every finding of every model of one cohort, with the screening value next to it."""
    parts = []
    for model in skill["model"].unique():
        entry = plan.get(model) or {}
        k = entry.get("interval")
        arm_k = f"restart_every{int(k)}" if k else None
        s = skill[skill["model"] == model]
        policies = set(s["policy"])
        comparisons = []
        if arm_k in policies:
            comparisons.append(("gain_over_never", {arm_k: "never"}))
            if "restart_every1" in policies and arm_k != "restart_every1":
                comparisons.append(("loss_to_daily", {arm_k: "restart_every1"}))
        own = cf.mechanism_arm(entry)
        if own in policies and arm_k in policies:
            comparisons.append(("mechanism_vs_restart", {own: arm_k}))
        for finding, against in comparisons:
            table = pr.paired(s, against, dmd, replicates, seed)
            if not table.empty:
                parts.append(table.assign(finding=finding))
    if not parts:
        return pd.DataFrame()
    table = pd.concat(parts, ignore_index=True)
    table = table.merge(screen, on=["model", "horizon_min", "finding"], how="left")
    table["replicated"] = [replicated(f, c, s if isinstance(s, str) else None)
                           for f, c, s in zip(table["finding"], table["verdict"], table["screen_verdict"])]
    return table


def replication_summary(replication: pd.DataFrame) -> pd.DataFrame:
    """Per cohort and finding: comparisons made, replicated, and the models that did not replicate."""
    if replication.empty:
        return pd.DataFrame()
    rows = []
    for (cohort, finding), g in replication.groupby(["cohort", "finding"], sort=False):
        known = g[g["replicated"].notna()]
        ok = known["replicated"].astype(bool)
        full = ~known["model"].map(is_twin)
        failed = known[~ok]
        rows.append({"cohort": cohort, "finding": finding, "comparisons": int(len(known)),
                     "replicated": int(ok.sum()), "full_models": f"{int(ok[full].sum())} of {int(full.sum())}",
                     "not_replicated": ", ".join(f"{m} {int(h)}" for m, h in zip(failed["model"], failed["horizon_min"])),
                     "without_screening_value": int(g["replicated"].isna().sum())})
    return pd.DataFrame(rows)


def low_table(metrics: pd.DataFrame) -> pd.DataFrame:
    """Sensitivity and precision of the forecast lows, and their change from never, per model, arm and horizon."""
    columns = [c for c in metrics.columns if c.startswith("low")]
    if not columns:
        return pd.DataFrame()
    t = metrics[["cohort", "period", "model", "policy", "horizon_min", "n"] + columns].copy()
    never = t[t["policy"] == "never"].set_index(["model", "horizon_min"])
    for c in columns:
        if c.endswith(("_sensitivity_pct", "_precision_pct")):
            t[f"{c}_vs_never"] = [v - never[c].get((m, h), np.nan)
                                  for v, m, h in zip(t[c], t["model"], t["horizon_min"])]
    return t


# ============================================================================ figures


def plot_replication(replication: pd.DataFrame, models: list[str], cohorts: list[str]):
    """Per finding and full model: the screening and every cohort (difference in skill points, mean over horizons)."""
    t = replication[replication["model"].isin(models)]
    findings = [f for f in FINDINGS if f in set(t["finding"])]
    fig, axes = model_axes(len(findings), 5.6, 3.4, ncols=len(findings))
    for ax, finding in zip(axes, findings):
        f = t[t["finding"] == finding]
        shown = [m for m in models if m in set(f["model"])]
        for i, model in enumerate(shown):
            m = f[f["model"] == model]
            screen = m.groupby("horizon_min")["screen_difference"].first().mean()
            ax.scatter(100 * screen, i + 0.3, color=TEXT_SECONDARY, marker="D", s=18,
                       label=SCREEN if i == 0 else None, zorder=3)
            for j, cohort in enumerate(c for c in cohorts if c in set(m["cohort"])):
                c = m[m["cohort"] == cohort]
                y = i - 0.3 + 0.5 * j / max(1, len(cohorts) - 1)
                ax.plot([100 * c["low"].mean(), 100 * c["high"].mean()], [y, y], color=color(model), lw=1.0,
                        alpha=0.5)
                known = c["replicated"].dropna()
                ax.scatter(100 * c["skill_difference"].mean(), y, color=color(model), s=12,
                           marker="s" if known.empty else "o" if known.astype(bool).all() else "x")
        ax.axvline(0, color=TEXT_SECONDARY, lw=0.8)
        ax.set_yticks(range(len(shown)), [label(m) for m in shown], fontsize=7)
        ax.set_title(FINDINGS[finding], loc="left", fontsize=8)
        ax.set_xlabel("skill points (mean over horizons)")
    if findings:
        fig.legend(*axes[0].get_legend_handles_labels(), fontsize=7, loc="upper right", frameon=False)
    fig.suptitle("Confirmation (D92): each finding of the screening (diamond) and in every cohort (circle: replicated "
                 "at every horizon; cross: not; square: no screening value; cohorts from top to bottom in config "
                 "order)", fontweight="bold", x=0.01, ha="left", fontsize=9)
    fig.tight_layout()
    return fig


def plot_weekly(weekly: pd.DataFrame, models: list[str], cohorts: list[str], plan: dict, horizon: int = 60):
    """Per cohort, the weekly gain over never of the refit at the optimal interval (60 min)."""
    w = weekly[weekly["horizon_min"] == horizon]
    shown = [c for c in cohorts if c in set(w["cohort"])]
    fig, axes = model_axes(len(shown), 4.6, 3.0)
    for ax, cohort in zip(axes, shown):
        c = w[w["cohort"] == cohort]
        present = set(c["model"])  # a cohort without insulin shows the glucose-only twins
        for model in [m if m in present else twin_of(m, FAMILIES) for m in models]:
            k = plan.get(model, {}).get("interval")
            m = c[(c["model"] == model) & (c["policy"] == f"restart_every{int(k)}")] if k else pd.DataFrame()
            if m.empty:
                continue
            m = m.sort_values("week")
            ax.plot(m["week"], 100 * m["gain_mean"], color=color(model), lw=1.3, label=label(model))
            ax.fill_between(m["week"], 100 * m["gain_low"], 100 * m["gain_high"], color=color(model), alpha=0.12,
                            lw=0)
        ax.axhline(0, color=TEXT_SECONDARY, lw=0.8)
        ax.set_title(cohort, loc="left", fontsize=9)
        ax.set_xlabel("weeks since the anchor")
        ax.set_ylabel("daily gain over never (points)")
    if shown:
        axes[0].legend(fontsize=6, loc="upper left")
    fig.suptitle(f"Gain of refitting at the optimal interval over never, week by week ({horizon} min; 95 % interval; "
                 f"cohorts without insulin: the glucose-only twins)",
                 fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    return fig


def plot_exog(exog: pd.DataFrame, cohorts: list[str], plan: dict):
    """Per cohort, what insulin and carbohydrates add to the gain of refitting at the optimal interval."""
    rows = []
    for r in exog.itertuples():
        k = plan.get(r.model, {}).get("interval")
        if k and r.policy == f"restart_every{int(k)}":
            rows.append(r)
    t = pd.DataFrame(rows)
    shown = [c for c in cohorts if not t.empty and c in set(t["cohort"])]
    fig, axes = model_axes(len(shown), 4.6, 3.0)
    for ax, cohort in zip(axes, shown):
        c = t[t["cohort"] == cohort]
        models = list(dict.fromkeys(c["model"]))
        for i, model in enumerate(models):
            m = c[c["model"] == model].sort_values("horizon_min")
            for j, r in enumerate(m.itertuples()):
                y = i + (j - 1) * 0.22
                ax.plot([100 * r.low, 100 * r.high], [y, y], color=color(model), lw=1.2, alpha=0.4 + 0.3 * j)
                ax.scatter(100 * r.difference, y, color=color(model), s=14, alpha=0.4 + 0.3 * j,
                           label=f"{int(r.horizon_min)} min" if i == 0 else None)
        ax.axvline(0, color=TEXT_SECONDARY, lw=0.8)
        ax.set_yticks(range(len(models)), [label(m) for m in models], fontsize=7)
        ax.set_title(cohort, loc="left", fontsize=9)
        ax.set_xlabel("gain with every input - glucose only (points)")
    if shown:
        axes[0].legend(fontsize=6, loc="lower right")
    fig.suptitle("Insulin and carbohydrates (D90): how much more refitting gains with them, per cohort",
                 fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    return fig


def plot_lows(lows: pd.DataFrame, models: list[str], cohort: str, plan: dict, horizon: int = 30):
    """Sensitivity and precision below 70 mg/dL of never and of the refit at the optimal interval (main cohort)."""
    t = lows[(lows["cohort"] == cohort) & (lows["horizon_min"] == horizon) & lows["model"].isin(models)]
    fig, axes = model_axes(2, 5.0, 3.0, ncols=2)
    for ax, measure in zip(axes, ("low70_sensitivity_pct", "low70_precision_pct")):
        shown = [m for m in models if m in set(t["model"])]
        for i, model in enumerate(shown):
            k = plan.get(model, {}).get("interval")
            for j, policy in enumerate(["never"] + ([f"restart_every{int(k)}"] if k else [])):
                v = t[(t["model"] == model) & (t["policy"] == policy)][measure]
                if v.empty:
                    continue
                ax.bar(i + (j - 0.5) * 0.38, float(v.iloc[0]), width=0.36, color=color(model),
                       alpha=0.45 if policy == "never" else 0.95, edgecolor="white", lw=0.4)
        ax.set_xticks(range(len(shown)), [label(m) for m in shown], fontsize=7, rotation=20, ha="right")
        ax.set_ylabel("%")
        ax.set_title(measure.replace("low70_", "below 70 mg/dL: ").replace("_pct", ""), loc="left", fontsize=9)
    fig.suptitle(f"Forecast lows in {cohort}, {horizon} min (light: never; dark: refit at the optimal interval)",
                 fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    return fig


# ============================================================================ summary


def write_summary(replication, tally, exog, growth, lows, fit_rows, plan, cohorts, models, manifest, release,
                  resources) -> str:
    lines = ["# 51 - Confirmation analysis", "", f"Code release {release}.", f"Machine: {resources.describe()}",
             f"Cohorts {cohorts}; models {models}; runs of script 50 (release {manifest.get('package_version')}). "
             f"Screening values: scripts 31 and 41 on the screen group of the core dataset. Comparisons and rules "
             f"fixed before the confirmation (D92).", ""]
    problems = [f"{r.cohort} {r.model}: {r.error}" for r in fit_rows.itertuples() if r.error] \
        if not fit_rows.empty else []
    lines += ["## Warnings", ""] + ([f"- {p}" for p in problems] or ["- none"]) + [""]
    if not tally.empty:
        lines += ["## Replication (comparisons replicated of those made; every horizon and model, twins included)", "",
                  "Rules: " + "; ".join(f"`{k}`: {v}" for k, v in FINDINGS.items()) + ".", "",
                  markdown_table(tally), ""]
    if not replication.empty:
        r = replication.copy()
        r["screen"] = [f"{100 * d:+.2f} ({v})" if pd.notna(d) else "" for d, v in
                       zip(r["screen_difference"], r["screen_verdict"])]
        r["confirmation"] = [f"{100 * d:+.2f} ({100 * lo:+.2f} to {100 * hi:+.2f}; {v})" for d, lo, hi, v in
                             zip(r["skill_difference"], r["low"], r["high"], r["verdict"])]
        r["replicated"] = r["replicated"].map({True: "yes", False: "no"}).fillna("no screening value")
        full = r[~r["model"].map(is_twin)]
        lines += ["## Full models: screening and confirmation (skill points)", "",
                  markdown_table(full[["cohort", "finding", "model", "policy", "against", "horizon_min", "patients",
                                       "screen", "confirmation", "replicated"]]), "",
                  "The twins are in `replication.csv`.", ""]
    if not exog.empty:
        e = exog.copy()
        e = e[[p == f"restart_every{int(plan.get(m, {}).get('interval') or 0)}" for p, m in zip(e["policy"], e["model"])]]
        for col in ("gain_all_inputs", "gain_glucose_only", "difference", "low", "high"):
            e[col] = (100 * e[col]).round(2)
        lines += ["## Insulin and carbohydrates (D90): gain over never of the refit at the optimal interval (points)",
                  "", markdown_table(e[["cohort", "model", "horizon_min", "patients", "gain_all_inputs",
                                        "gain_glucose_only", "difference", "low", "high", "verdict"]]), "",
                  "Every arm is in `exog_confirmation.csv`.", ""]
    if not growth.empty:
        g = growth[[p == f"restart_every{int(plan.get(m, {}).get('interval') or 0)}" and h == 60
                    for p, m, h in zip(growth["policy"], growth["model"], growth["horizon_min"])]].copy()
        if not g.empty:
            for col in ("gain_at_anchor", "growth_per_week", "growth_low", "growth_high"):
                g[col] = (100 * g[col]).round(3)
            lines += ["## Growth of the gain over never (60 min, refit at the optimal interval; points per week)", "",
                      markdown_table(g[["cohort", "model", "patients", "days", "gain_at_anchor", "growth_per_week",
                                        "growth_low", "growth_high", "spec"]]), ""]
    if not lows.empty:
        show = [c for c in ("cohort", "model", "policy", "horizon_min", "low70_share_pct", "low70_sensitivity_pct",
                            "low70_precision_pct", "low54_sensitivity_pct", "low54_precision_pct") if c in lows]
        main = lows[(lows["cohort"] == cohorts[0]) & (lows["horizon_min"] == 30) & ~lows["model"].map(is_twin)]
        lines += [f"## Forecast lows in {cohorts[0]} (30 min, %)", "", markdown_table(main[show].round(1)), "",
                  "Every cohort, arm and horizon is in `low_glucose.csv`.", ""]
    lines += ["## Checks to look at", "",
              "- `replication.png`: which findings of the screening hold in every cohort.",
              "- `weekly_gain.png`: whether the gain of refitting keeps growing (Loop: up to 36 weeks).",
              "- `exog_confirmation.png`: whether insulin and carbohydrates add to the gain outside the core dataset.",
              "- `low_glucose.png`: whether refitting forecasts the lows better.", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
