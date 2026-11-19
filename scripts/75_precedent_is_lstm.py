"""75 - The closest precedent: the incrementally retrained stacked LSTM of Shen and Kleinberg (2025) against the
refitting the study recommends (HANDOFF 7.10, D103).

IS-LSTM continues the personal LSTM every 7 days on the last 7 days only; with
its "parameter transfer" it skips a week whose glucose a two-sample t-test
cannot tell from the week before. Here it runs with the confirmation's
machinery (``sim.cohort_runs``) on the confirmation's patients, anchor (day 14)
and test days, as arms of ``sim.mechanisms``:

- ``never``: the fit at the anchor, shared by every arm;
- ``incremental_w7_every7``: IS-LSTM without parameter transfer;
- ``incremental_w7_t0.05_every7``: IS-LSTM with it (Welch t-test, p > 0.05
  keeps the model in use);

for the LSTM of the study and its glucose-only twin (IS-LSTM did best on
glucose alone). Against them, read from script 50 on the same patients and
days: the restart at the optimal interval (every 3 days), the recommended
mechanism (continuing the model every 3 days on the growing window, D91) and,
in REPLACE-BG, the restart every day.

Differences from the original, by design: the starting model is the study's
(population model fine-tuned on 14 days), not a stacked LSTM trained on the
first days; early stopping uses the last 2 days of the window, because the
original validated on the next day, which the protocol does not see yet; and
the network is the study's LSTM. What is compared is the policy.

Measures: skill of every arm; paired differences (D75 verdicts with the LSTM's
DMD); the share of weekly refits the t-test skipped; computing seconds per test
day; the daily gain over never by week. A check: the never of this script
against that of script 50 (the same fit at the anchor; a GPU can make them
differ slightly). Rule R6: RMSE and Clarke zones of every arm, with the grid
data.

Inputs:
    <derived_data>/clean/<dataset>/anchors.parquet; <derived_data>/features/<dataset>/grid/  (scripts 04, 05)
    <derived_data>/runs/31_.../confirmation.yaml, <derived_data>/runs/41_.../mechanisms.yaml
    <derived_data>/models/<core>/population/   (script 11)
    <derived_data>/runs/50_confirmation_runs/day_metrics.parquet; results/50_confirmation_runs/arm_overview.csv
    results/12_stage0_population_transfer/noise_floor.csv   (DMD, D75)

Derived data:
    <derived_data>/runs/75_precedent_is_lstm/cache/...; <derived_data>/runs/75_precedent_is_lstm/day_metrics.parquet

Outputs (results/75_precedent_is_lstm/):
    skill.csv               per cohort, model, arm and horizon: patients, skill (95 %), RMSE
    versus.csv              paired per patient: each IS-LSTM arm against never, the restart, the recommended
                            mechanism and the other IS-LSTM arm; this never against script 50's (DMD verdicts)
    refits.csv              per cohort, model and arm: weekly refits run and skipped, computing seconds per test day
    weekly_gain.csv         daily gain over never by week since the anchor (patient means, 95 %)
    metrics_overall.csv     pooled forecasts (R6: RMSE, MAE, MARD, bias, Clarke zones) and forecast lows
    clarke_zones.csv / clarke_histogram.csv / clarke_points_sample.csv   (R6, the arms run here)
    fit_log.csv / task_log.csv / jobs.csv
    precedent.png, weekly_gain.png
    summary.md, run_manifest.json

Usage:
    python scripts/75_precedent_is_lstm.py
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

from t1dretrain.compute import detect_resources, parallel_map, share_gpus  # noqa: E402
from t1dretrain.config import CONFIG_DIR, derived_data_dir, load_project_config  # noqa: E402
from t1dretrain.data.splits import cohort_rule  # noqa: E402
from t1dretrain.eval import confirmation as cf  # noqa: E402
from t1dretrain.eval import policy_analysis as pa  # noqa: E402
from t1dretrain.eval import policy_report as pr  # noqa: E402
from t1dretrain.eval.stage0_report import SUMS, DMDTable, finish, label, read_noise_floor  # noqa: E402
from t1dretrain.eval.stats import bootstrap_mean  # noqa: E402
from t1dretrain.outputs import ScriptOutput, markdown_table  # noqa: E402
from t1dretrain.plotting import TEXT_SECONDARY, model_axes, set_style  # noqa: E402
from t1dretrain.sim import cohort_runs as cr  # noqa: E402
from t1dretrain.sim.mechanisms import parse_arm  # noqa: E402
from t1dretrain.sim.stage0 import FAMILIES, split_name  # noqa: E402

SCRIPT_ID = "75_precedent_is_lstm"
SOURCE = "50_confirmation_runs"
STUDY = "study: "  # prefix of the arms read from script 50
ARM_LABELS = {"incremental_w7_every7": "IS-LSTM", "incremental_w7_t0.05_every7": "IS-LSTM, parameter transfer"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--paths", type=Path, default=CONFIG_DIR / "paths.yaml", help="paths file (config/paths.yaml)")
    parser.add_argument("--patients", type=int, default=None, help="only the first N patients per cohort (check)")
    parser.add_argument("--recompute", action="store_true", help="ignore the cached results of earlier runs")
    parser.add_argument("--skip-code-check", action="store_true",
                        help="run even if the code differs from the released manifest")
    return parser.parse_args()


def arm_label(arm: str) -> str:
    if arm.startswith(STUDY):
        return arm
    return ARM_LABELS.get(arm, arm)


# ============================================================================ main


def main() -> None:
    import json
    import time

    run_start = time.time()
    args = parse_args()
    release = verify_code(strict=not args.skip_code_check)
    config = load_project_config()
    resources = detect_resources(config.get("compute", {}).get("reserve_cores"))
    print(resources.describe())
    print(f"GPU sharing: at most {share_gpus(config)} processes compute on one GPU at a time (D73)")
    out = ScriptOutput(SCRIPT_ID, resources=resources)
    set_style()
    derived = derived_data_dir(args.paths)
    results = Path(out.dir).parent
    core = config.get("core_dataset", "replace_bg")
    conf, cfg = config["confirmation"], config["precedent"]
    seed, replicates = int(config["random_seed"]), int(config["stage0"]["bootstrap_replicates"])
    stage2 = config.get("stage2_analysis", {})
    min_origins, clip = int(stage2.get("min_origins_per_day", 12)), tuple(stage2.get("skill_clip", [-1.0, 1.0]))
    arms = [str(a) for a in cfg["arms"]]
    for arm in arms:
        parse_arm(arm)  # an unknown arm stops the run here
    population_dir = derived / "models" / core / "population"
    choices = cr.read_choices(population_dir)
    base = cr.base_settings(config, choices, population_dir)
    policies_doc, mechanisms_doc = cr.read_plan(derived)
    models, plan, notes = cr.plan_models(policies_doc, mechanisms_doc, list(cfg["models"]), population_dir,
                                         base["seeds"])
    by_key = {cf.cohort_key(c["dataset"], c["cohort"]): c for c in conf["cohorts"]}
    unknown = [k for k in cfg["cohorts"] if k not in by_key]
    if unknown:
        raise SystemExit(f"precedent.cohorts: {unknown} are not cohorts of the confirmation")

    runs = derived / "runs" / SCRIPT_ID
    recompute = run_start if args.recompute else None
    jobs, rows, all_anchors, arms_of, test_days_of = [], [], [], {}, {}
    for order, key in enumerate(cfg["cohorts"]):
        c = by_key[key]
        rule = cohort_rule(c["dataset"], c["cohort"])
        test_days = int(c.get("test_days", conf["test_days"]))
        settings = {**base, "test_days": test_days}
        anchors, counts = cr.cohort_patients(c, derived, [int(a) for a in conf["anchor_days"]], seed, args.patients)
        runnable = cf.models_for_rule(models, rule)
        arms_of[key] = {m: arms for m in runnable}
        test_days_of[key] = test_days
        rows.append({"cohort": key, "usable_rule": rule, **counts, "test_days": test_days,
                     "models": " ".join(runnable)})
        if anchors.empty or not runnable:
            continue
        all_anchors.append(anchors.assign(cohort=key))
        jobs += cr.cohort_jobs(key, c["dataset"], rule, order, anchors, arms_of[key], settings, choices, runs,
                               recompute, derived, cr.device_for(resources), stream0=len(jobs))
    cohort_table = pd.DataFrame(rows)
    print(cohort_table.to_string(index=False))
    jobs.sort(key=cr.job_order)
    print(f"Models {models}; arms {arms}; {len(jobs)} jobs")
    statuses = parallel_map(cr.run_job, jobs, resources.n_workers, label="IS-LSTM jobs")
    own, info, accumulator, missing = cr.collect(jobs)
    runs.mkdir(parents=True, exist_ok=True)
    if not own.empty:
        own.to_parquet(runs / "day_metrics.parquet", index=False)
    fit_log, task_log = pd.DataFrame(info), cr.task_table(statuses)
    anchors_all = pd.concat(all_anchors, ignore_index=True) if all_anchors else pd.DataFrame()

    study, mismatch = study_arms(derived, own, models, plan, list(cfg["cohorts"]))
    days = pd.concat([own, study], ignore_index=True) if not study.empty else own
    dmd = DMDTable(read_noise_floor(results / "12_stage0_population_transfer" / "noise_floor.csv"),
                   [m for m in models if split_name(m)[0] in FAMILIES])
    skill = skill_table(days) if not days.empty else pd.DataFrame()
    means = mean_table(skill, replicates, seed) if not skill.empty else pd.DataFrame()
    versus = comparisons(skill, plan, arms, dmd, replicates, seed) if not skill.empty else pd.DataFrame()
    refits = refit_table(fit_log, results / SOURCE / "arm_overview.csv", plan, test_days_of)
    weekly = weekly_table(days, arms, plan, min_origins, clip) if not days.empty else pd.DataFrame()
    metrics = pooled(own) if not own.empty else pd.DataFrame()

    out.csv(means.round(5), "skill")
    out.csv(versus.round(5), "versus")
    out.csv(refits.round(4), "refits")
    out.csv(weekly.round(5), "weekly_gain")
    out.csv(metrics.round(4), "metrics_overall")
    out.csv(accumulator.zones(), "clarke_zones")
    out.csv(accumulator.histogram(), "clarke_histogram")
    out.csv(accumulator.sample(), "clarke_points_sample")
    out.csv(fit_log, "fit_log")
    out.csv(task_log, "task_log")
    out.csv(anchors_all, "jobs")
    if not versus.empty:
        out.figure(plot_versus(versus), "precedent")
    if not weekly.empty:
        out.figure(plot_weekly(weekly), "weekly_gain")
    out.text(write_summary(cohort_table, means, versus, refits, metrics, task_log, fit_log, missing, mismatch, notes,
                           models, arms, release, resources), "summary.md")
    out.manifest(settings={"models": models, "arms": arms, "cohorts": list(cfg["cohorts"]),
                           "plan": {m: plan[m] for m in models},
                           **{k: v for k, v in base.items() if k != "population_dir"}},
                 notes="IS-LSTM arms run here; the study's arms read from the day metrics of script 50. "
                       + json.dumps({"notes": notes}))
    print(f"Done. Results in {out.dir}")


# ============================================================================ tables


def study_arms(derived: Path, own: pd.DataFrame, models: list[str], plan: dict,
               cohorts: list[str]) -> tuple[pd.DataFrame, str]:
    """The arms of script 50 on the same cohorts, models and patients (named ``study: <arm>``), and a check that
    the persistence here has the origins and errors of script 50."""
    path = derived / "runs" / SOURCE / "day_metrics.parquet"
    if not path.exists() or own.empty:
        return pd.DataFrame(), f"{path} not found: no arms of the study to compare with"
    days = pd.read_parquet(path)
    days = days[days["cohort"].isin(cohorts) & days["patient_id"].isin(set(own["patient_id"]))]
    keep = days["model"].isin(models) & days["policy"].isin(
        {"never", "restart_every1"} | {a for m in models for a in (cf.recommended_arm(plan[m]),
                                                                   f"restart_every{plan[m]['interval']}") if a})
    study = days[keep].copy()
    study["policy"] = STUDY + study["policy"]
    keys = ["cohort", "patient_id", "horizon_min", "day"]
    a = own[own["model"] == "persistence"].drop_duplicates(keys).set_index(keys)[["n", "sse"]]
    b = days[days["model"] == "persistence"].drop_duplicates(keys).set_index(keys)[["n", "sse"]]
    both = a.join(b, how="inner", lsuffix="_75", rsuffix="_50")
    differ = (both["n_75"] != both["n_50"]) | ~np.isclose(both["sse_75"], both["sse_50"], rtol=1e-6)
    mismatch = "" if not differ.any() else (f"{int(differ.sum())} of {len(both)} patient-days of persistence differ "
                                            "from script 50 (other origins)")
    if len(both) < len(a):
        mismatch += f"; {len(a) - len(both)} patient-days of persistence here are not in script 50"
    return study, mismatch.strip("; ")


def skill_table(days: pd.DataFrame) -> pd.DataFrame:
    """Per cohort, period, model, arm, horizon and patient: pooled RMSE and skill against persistence (the one run
    here; the study's arms are on the same origins)."""
    own_persistence = days[(days["model"] == "persistence") & ~days["policy"].astype(str).str.startswith(STUDY)]
    d = pd.concat([days[days["model"] != "persistence"], own_persistence], ignore_index=True)
    parts = [pr.patient_skill(g).assign(cohort=cohort) for cohort, g in d.groupby("cohort", sort=False)]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def mean_table(skill: pd.DataFrame, replicates: int, seed: int) -> pd.DataFrame:
    rows = []
    for (cohort, model, policy, horizon), g in skill.groupby(["cohort", "model", "policy", "horizon_min"],
                                                             sort=False):
        mean, low, high = bootstrap_mean(g["skill"].to_numpy(), replicates, seed)
        rows.append({"cohort": cohort, "model": model, "arm": policy, "horizon_min": int(horizon),
                     "patients": int(g["patient_id"].nunique()), "skill": mean, "low": low, "high": high,
                     "rmse": float(g["rmse"].mean())})
    return pd.DataFrame(rows)


def comparisons(skill: pd.DataFrame, plan: dict, arms: list[str], dmd: DMDTable, replicates: int,
                seed: int) -> pd.DataFrame:
    """Each IS-LSTM arm against never (this script's), the restart at the optimal interval and the recommended
    mechanism (script 50's) and the other IS-LSTM arm; this script's never against script 50's."""
    incremental = [a for a in arms if parse_arm(a)[0] == "incremental"]
    rows = []
    for (cohort, model), s in skill.groupby(["cohort", "model"], sort=False):
        entry = plan.get(model) or {}
        restart = f"{STUDY}restart_every{entry.get('interval')}"
        recommended = STUDY + str(cf.recommended_arm(entry))
        pairs = [(a, b, kind) for a in incremental for b, kind in
                 (("never", "against never"), (restart, "against the restart"),
                  (recommended, "against the recommended mechanism"))]
        pairs += [(b, a, "parameter transfer") for a, b in zip(incremental, incremental[1:])]
        pairs.append(("never", STUDY + "never", "check: never here against script 50"))
        for a, b, kind in pairs:
            table = pr.paired(s, {a: b}, dmd, replicates, seed)
            if not table.empty:
                rows.append(table.assign(cohort=cohort, comparison=kind))
    if not rows:
        return pd.DataFrame()
    table = pd.concat(rows, ignore_index=True)
    first = ["cohort", "comparison", "model", "horizon_min", "policy", "against"]
    return table[first + [c for c in table.columns if c not in first]]


def refit_table(fit_log: pd.DataFrame, overview: Path, plan: dict, test_days_of: dict) -> pd.DataFrame:
    """Per cohort, model and arm: weekly refits run and skipped (IS-LSTM arms) and computing seconds per test day;
    the study's arms from script 50's overview."""
    rows = []
    if not fit_log.empty:
        log = fit_log.copy()
        log["skipped"] = log["skipped"].eq(True) if "skipped" in log else False  # missing (no IS-LSTM arm) = False
        for (cohort, model, policy), g in log.groupby(["cohort", "model", "policy"], sort=False):
            refits = g[g["weighting"] == policy]
            cost = pr.cost_table(g, int(test_days_of.get(cohort, 1)))
            rows.append({"cohort": cohort, "model": model, "arm": policy, "source": "75",
                         "patients": int(g["patient_id"].nunique()),
                         "refits_per_patient": float(len(refits) / max(1, g["patient_id"].nunique())),
                         "skipped_share": float(refits["skipped"].mean()) if len(refits) else np.nan,
                         "seconds_per_day": float(cost["seconds_per_day"].iloc[0]) if len(cost) else np.nan})
    if overview.exists():
        o = pd.read_csv(overview)
        for model, entry in plan.items():
            wanted = {"never", f"restart_every{entry.get('interval')}", str(cf.recommended_arm(entry))}
            for r in o[(o["model"] == model) & o["arm"].isin(wanted) & o["cohort"].isin(test_days_of)].itertuples():
                rows.append({"cohort": r.cohort, "model": model, "arm": STUDY + r.arm, "source": "50",
                             "patients": int(r.patients), "refits_per_patient": float(r.fits_per_anchor) - 1,
                             "skipped_share": np.nan, "seconds_per_day": float(r.seconds_per_day)})
    return pd.DataFrame(rows)


def weekly_table(days: pd.DataFrame, arms: list[str], plan: dict, min_origins: int, clip) -> pd.DataFrame:
    """The daily gain over never (this script's) by week since the anchor, of every arm here and of the study's
    recommended mechanism."""
    parts = []
    for cohort, d in days.groupby("cohort", sort=False):
        d = d[~((d["model"] == "persistence") & d["policy"].astype(str).str.startswith(STUDY))]
        refits = [a for a in arms if a != "never"] + sorted({STUDY + str(cf.recommended_arm(plan[m]))
                                                             for m in plan if m in set(d["model"])})
        gain = pa.daily_gain(d, refits, "never", min_origins, clip)
        if not gain.empty:
            parts.append(pa.weekly_gain(gain).assign(cohort=cohort))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def pooled(own: pd.DataFrame) -> pd.DataFrame:
    sums = SUMS + cf.low_sums(own)
    t = own.groupby(["cohort", "period", "model", "policy", "horizon_min"], as_index=False)[sums].sum()
    return cf.with_lows(finish(t))


# ============================================================================ figures


def plot_versus(versus: pd.DataFrame):
    """Per cohort and model: each IS-LSTM arm minus the recommended mechanism and minus never (points, 95 %)."""
    v = versus[versus["comparison"].isin(["against never", "against the recommended mechanism"])]
    groups = list(dict.fromkeys(zip(v["cohort"], v["model"])))
    fig, axes = model_axes(len(groups), 4.0, 2.6, ncols=min(3, max(1, len(groups))))
    for ax, (cohort, model) in zip(axes, groups):
        g = v[(v["cohort"] == cohort) & (v["model"] == model)]
        for k, ((policy, kind), h) in enumerate(g.groupby(["policy", "comparison"], sort=False)):
            h = h.sort_values("horizon_min")
            x = np.arange(len(h)) + (k - 1.5) * 0.12
            ax.errorbar(x, 100 * h["skill_difference"], yerr=[100 * (h["skill_difference"] - h["low"]),
                                                              100 * (h["high"] - h["skill_difference"])],
                        fmt="o" if "never" in kind else "s", ms=4, lw=0.9,
                        mfc="white" if "parameter" in arm_label(policy) else None,
                        label=f"{arm_label(policy)} {kind.replace('against the ', 'vs ').replace('against ', 'vs ')}")
            ax.set_xticks(range(len(h)), [f"{int(m)} min" for m in h["horizon_min"]])
        ax.axhline(0, color=TEXT_SECONDARY, lw=0.8)
        ax.set_ylabel("skill difference (points)")
        ax.set_title(f"{cohort}, {label(model)}", loc="left", fontsize=8)
    axes[0].legend(fontsize=5.5)
    fig.suptitle("IS-LSTM (Shen and Kleinberg 2025) against never and the recommended refitting", fontweight="bold",
                 x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    return fig


def plot_weekly(weekly: pd.DataFrame, horizon: int = 60):
    """Per cohort: the daily gain over never by week, at ``horizon``, of the full LSTM."""
    w = weekly[(weekly["horizon_min"] == horizon) & ~weekly["model"].str.endswith("_noexog")]
    cohorts = list(dict.fromkeys(w["cohort"]))
    fig, axes = model_axes(len(cohorts), 4.0, 2.6, ncols=min(3, max(1, len(cohorts))))
    for ax, cohort in zip(axes, cohorts):
        for policy, g in w[w["cohort"] == cohort].groupby("policy", sort=False):
            g = g.sort_values("week")
            ax.plot(g["week"] + 1, 100 * g["gain_mean"], marker="o", ms=2.5, lw=1.2, label=arm_label(policy))
            ax.fill_between(g["week"] + 1, 100 * g["gain_low"], 100 * g["gain_high"], alpha=0.1, lw=0)
        ax.axhline(0, color=TEXT_SECONDARY, lw=0.8)
        ax.set_xlabel("week since the anchor")
        ax.set_ylabel("gain over never (points)")
        ax.set_title(cohort, loc="left", fontsize=8)
    axes[0].legend(fontsize=6)
    fig.suptitle(f"Gain over never by week ({horizon} min), LSTM", fontweight="bold", x=0.01, ha="left",
                 fontsize=10)
    fig.tight_layout()
    return fig


# ============================================================================ summary


def write_summary(cohorts, means, versus, refits, metrics, task_log, fit_log, missing, mismatch, notes, models, arms,
                  release, resources) -> str:
    lines = ["# 75 - IS-LSTM (Shen and Kleinberg 2025) against the recommended refitting", "",
             f"Code release {release}.", f"Machine: {resources.describe()}",
             f"Models {models}; arms run here {arms}; the study's arms (prefix `{STUDY}`) read from script 50 on the "
             "same patients and days. D103.", ""]
    problems = list(notes)
    if not task_log.empty:
        failed = task_log[task_log["status"] == "error"].head(20)
        problems += [f"{r.job} failed: {r.error}" for r in failed.itertuples()]
    if not fit_log.empty and "error" in fit_log:
        for (model, error), g in fit_log[fit_log["error"].notna()].groupby(["model", "error"]):
            problems.append(f"{model}: {len(g)} fits failed: {error}")
    if missing:
        problems.append(f"{len(missing)} results missing (run the script again: what finished comes from the "
                        "cache): " + ", ".join(missing[:10]) + (" ..." if len(missing) > 10 else ""))
    problems += [mismatch] if mismatch else []
    lines += ["## Warnings", ""] + ([f"- {p}" for p in problems] or ["- none"]) + [""]
    lines += ["## Cohorts", "", markdown_table(cohorts.drop(columns=["models"], errors="ignore")), ""]
    if not means.empty:
        m = means.copy()
        m["value"] = [f"{100 * s:.2f} ({100 * lo:.2f} to {100 * hi:.2f})" for s, lo, hi in
                      zip(m["skill"], m["low"], m["high"])]
        m["arm"] = m["arm"].map(arm_label)
        table = m.pivot_table(index=["cohort", "model", "arm"], columns="horizon_min", values="value",
                              aggfunc="first")
        table.columns = [f"skill {h} min (%)" for h in table.columns]
        lines += ["## Skill (mean over patients, 95 % bootstrap interval)", "", markdown_table(table.reset_index()),
                  ""]
    if not versus.empty:
        v = versus.copy()
        v["cell"] = [f"{100 * d:+.2f} ({100 * lo:+.2f} to {100 * hi:+.2f}; {verdict})" for d, lo, hi, verdict in
                     zip(v["skill_difference"], v["low"], v["high"], v["verdict"])]
        v["policy"], v["against"] = v["policy"].map(arm_label), v["against"].map(arm_label)
        table = v.pivot_table(index=["cohort", "model", "comparison", "policy", "against"], columns="horizon_min",
                              values="cell", aggfunc="first")
        table.columns = [f"{h} min" for h in table.columns]
        lines += ["## Paired comparisons (skill points; tie below the LSTM's DMD)", "",
                  markdown_table(table.reset_index()), ""]
    if not refits.empty:
        r = refits.copy()
        r["arm"] = r["arm"].map(arm_label)
        lines += ["## Refits and cost (refits per patient after the anchor; computing seconds per test day)", "",
                  markdown_table(r.round(3)), ""]
    if not metrics.empty:
        show = [c for c in ("cohort", "model", "policy", "horizon_min", "n", "rmse", "clarke_AB_pct", "clarke_D_pct",
                            "low70_sensitivity_pct", "low70_precision_pct") if c in metrics]
        lines += ["## Pooled forecasts of the arms run here (R6)", "", markdown_table(metrics[show].round(3)), ""]
    lines += ["## Checks to look at", "",
              "- The never of this script against that of script 50: a tie (the same fit at the anchor).",
              "- `precedent.png`: IS-LSTM against never and against the recommended refitting, per cohort.", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
