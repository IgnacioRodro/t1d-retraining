"""72 - Recovery after real changes: how a model trained before a change fares after it, and how fast refitting
catches up (HANDOFF 7.10, D97).

Real changes come from script 04 (``change_events``): DCLP3's switch to closed
loop (Control-IQ) the day after randomisation, REPLACE-BG's randomisation,
BrisT1D's switches of loop and of sensor. The first day after the change is the
anchor: every model of the confirmation (D92, with its twin) is trained on the
``window_days`` days before it and follows the confirmation's arms over the
``recovery.test_days`` days after it (never, the refit at the optimal interval
and every day, its mechanism; ARX also online). Nothing is chosen here.

Groups (``recovery.groups``): DCLP3's switch (CLC arm) with, as a control,
DCLP3's SAP arm at randomisation (the same calendar, no change of loop), both
with the glucose-only models (insulin before the change is mostly missing);
REPLACE-BG's randomisation in the confirm group, every model; BrisT1D's
changes, every model, shown event by event (few). Events need
``min_days_each_side`` valid days on each side and a 5-min grid.

Analysis (``eval.recovery``): the daily skill against persistence, week by
week, of every arm; the gain of every refit over never week by week; the
change against its control (never's skill and the gains, difference of the
group means with a bootstrap interval); for DCLP3's switch, every arm against
the same model trained after the change (script 50's anchor, 14 days into
closed loop) on the same patients and days. Results are cached per group,
model, arm and event (``sim.cohort_runs``).

Inputs:
    <derived_data>/clean/<dataset>/change_events.parquet, patients.parquet   (script 04)
    <derived_data>/runs/31_.../confirmation.yaml, <derived_data>/runs/41_.../mechanisms.yaml
    <derived_data>/models/<core>/population/ (script 11); <derived_data>/features/<dataset>/ (script 05)
    <derived_data>/runs/50_confirmation_runs/day_metrics.parquet   (the models trained after the change)

Derived data:
    <derived_data>/runs/72_change_recovery/day_metrics.parquet and cache/<group>/<model>/<arm>/<event>.pkl

Outputs (results/72_change_recovery/):
    events.csv              per group: events, those in the arm or group, with enough days, on a 5-min grid, models
    jobs.csv                every event (patient, day of the change, valid days on each side)
    weekly_skill.csv        per group, model, arm, horizon and week since the change: mean daily skill (95 %)
    weekly_gain.csv         per group, model, refit, horizon and week: mean daily gain over never (95 %)
    change_vs_control.csv   the change group against its control, over every week and week by week
    trained_after.csv       every arm against the same model trained after the change, by week
    case_series.csv         groups with few events: weekly skill of every arm, event by event
    metrics_by_group.csv    pooled forecasts (R6: RMSE, MAE, MARD, bias, Clarke zones) and forecast lows
    clarke_zones.csv / clarke_histogram.csv / clarke_points_sample.csv   (R6)
    fit_log.csv / task_log.csv
    weekly_gain.png, change_vs_control.png, trained_after.png
    summary.md, run_manifest.json

Usage:
    python scripts/72_change_recovery.py
    python scripts/72_change_recovery.py --groups dclp3_to_closed_loop dclp3_sap_control --patients 3   # quick
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
from matplotlib.ticker import MaxNLocator  # noqa: E402

from t1dretrain.compute import detect_resources, parallel_map, share_gpus  # noqa: E402
from t1dretrain.config import CONFIG_DIR, derived_data_dir, load_project_config  # noqa: E402
from t1dretrain.data.splits import USABLE_RULES, group_patients  # noqa: E402
from t1dretrain.data.store import ProcessedStore  # noqa: E402
from t1dretrain.eval import confirmation as cf  # noqa: E402
from t1dretrain.eval import recovery as rc  # noqa: E402
from t1dretrain.eval.stage0_report import SUMS, color, finish, label  # noqa: E402
from t1dretrain.outputs import ScriptOutput, markdown_table  # noqa: E402
from t1dretrain.plotting import TEXT_SECONDARY, model_axes, set_style  # noqa: E402
from t1dretrain.sim import cohort_runs as cr  # noqa: E402
from t1dretrain.sim.mechanisms import parse_arm  # noqa: E402
from t1dretrain.sim.stage0 import is_twin, split_variant  # noqa: E402

SCRIPT_ID = "72_change_recovery"
SOURCE = "50_confirmation_runs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--paths", type=Path, default=CONFIG_DIR / "paths.yaml", help="paths file (config/paths.yaml)")
    parser.add_argument("--groups", nargs="*", default=None, help="subset of recovery.groups (by name)")
    parser.add_argument("--models", nargs="*", default=None, help="subset of the models (full names; twins follow)")
    parser.add_argument("--patients", type=int, default=None, help="only the first N events of each group (quick)")
    parser.add_argument("--recompute", action="store_true", help="ignore the cached results of earlier runs")
    parser.add_argument("--skip-code-check", action="store_true",
                        help="run even if the code differs from the released manifest")
    return parser.parse_args()


def group_events(group: dict, derived: Path, min_days: int, first: int | None) -> tuple[pd.DataFrame, dict]:
    """The anchors of one group: its events with enough valid days on each side, on a 5-min grid."""
    store = ProcessedStore(group["dataset"], root=derived, stage="clean")
    events = store.read_extra("change_events")
    if events.empty:
        return pd.DataFrame(), {"events": 0}
    arms = store.read_patients().set_index("patient_id")["arm"] if group.get("arm") else pd.Series(dtype=object)
    members = group_patients(group["dataset"], group["group"], allow_confirm=True) if group.get("group") else None
    chosen, counts = rc.select_events(events, group, arms, members, min_days)
    on_grid = set(cr.on_core_grid(chosen["patient_id"].unique(), cr.grid_steps(group["dataset"], derived)))
    chosen = chosen[chosen["patient_id"].isin(on_grid)].sort_values(["patient_id", "day_index"])
    if first:
        chosen = chosen.head(first)
    counts["on_grid"] = int(len(chosen))
    anchors = pd.DataFrame({"patient_id": chosen["patient_id"], "period": "whole",
                            "anchor_day": chosen["day_index"].astype(int), "anchor": pd.to_datetime(chosen["day"]),
                            "from": chosen["from"], "to": chosen["to"],
                            "valid_days_before": chosen["valid_days_before"],
                            "valid_days_after": chosen["valid_days_after"]})
    return anchors.reset_index(drop=True), counts


# ============================================================================ main


def main() -> None:
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
    core = config.get("core_dataset", "replace_bg")
    cfg = config["recovery"]
    analysis = config.get("stage2_analysis", {})
    min_origins, clip = int(analysis.get("min_origins_per_day", 12)), tuple(analysis.get("skill_clip", [-1.0, 1.0]))
    seed, replicates = int(config["random_seed"]), int(config["stage0"]["bootstrap_replicates"])
    population_dir = derived / "models" / core / "population"
    choices = cr.read_choices(population_dir)
    base = cr.base_settings(config, choices, population_dir)
    settings = {**base, "test_days": int(cfg["test_days"])}
    forgetting = float((base.get("arx") or {}).get("forgetting", 1.0))
    policies_doc, mechanisms_doc = cr.read_plan(derived)
    models, plan, notes = cr.plan_models(policies_doc, mechanisms_doc, args.models, population_dir, base["seeds"])
    groups = [g for g in cfg["groups"] if not args.groups or g["name"] in args.groups]
    for g in groups:
        if g["rule"] not in USABLE_RULES:
            raise SystemExit(f"recovery group {g['name']}: unknown rule {g['rule']!r} (data.splits.USABLE_RULES)")

    runs = derived / "runs" / SCRIPT_ID
    recompute = run_start if args.recompute else None
    jobs, rows, all_anchors, arms_of = [], [], [], {}
    for order, g in enumerate(groups):
        anchors, counts = group_events(g, derived, int(cfg["min_days_each_side"]), args.patients)
        runnable = cf.models_for_rule(models, g["rule"])
        arms_of[g["name"]] = {m: cf.confirmation_arms(plan[m]["policies"], plan[m]["mechanism"], plan[m]["interval"],
                                                      bool(cfg.get("daily", True)), forgetting) for m in runnable}
        for arms in arms_of[g["name"]].values():
            for arm in arms:
                parse_arm(arm)
        rows.append({"group": g["name"], "dataset": g["dataset"], "kind": g["kind"], "arm": g.get("arm", ""),
                     "core_group": g.get("group", ""), "rule": g["rule"], **counts, "models": " ".join(runnable)})
        if anchors.empty:
            continue
        all_anchors.append(anchors.assign(group=g["name"]))
        jobs += cr.cohort_jobs(g["name"], g["dataset"], g["rule"], order, anchors, arms_of[g["name"]], settings,
                               choices, runs, recompute, derived, cr.device_for(resources), stream0=len(jobs))
    event_table = pd.DataFrame(rows)
    print(event_table.to_string(index=False))
    jobs.sort(key=cr.job_order)
    print(f"Models {models}; {len(jobs)} jobs")
    statuses = parallel_map(cr.run_job, jobs, resources.n_workers, label="recovery jobs")
    days, infos, clarke, missing = cr.collect(jobs)
    runs.mkdir(parents=True, exist_ok=True)
    if not days.empty:
        days.to_parquet(runs / "day_metrics.parquet", index=False)
    fit_log, task_log = pd.DataFrame(infos), cr.task_table(statuses)

    skill = rc.daily(days, min_origins, clip) if not days.empty else pd.DataFrame()
    weekly, gains, weekly_gain, contrast, after, cases, metrics = (pd.DataFrame(),) * 7
    if not skill.empty:
        weekly = rc.weekly_skill(skill)
        gains = rc.daily_gains(skill)
        weekly_gain = rc.weekly_gain(gains)
        contrast = change_contrast(skill, gains, cfg.get("contrast"), int(cfg["test_days"]), replicates, seed)
        after = trained_after(skill, cfg.get("trained_after"), derived, min_origins, clip, replicates, seed)
        cases = case_series(skill, event_table, int(cfg.get("case_events", 5)))
        sums = SUMS + cf.low_sums(days)
        pooled = days.groupby(["cohort", "model", "policy", "horizon_min"], as_index=False)[sums].sum()
        metrics = cf.with_lows(finish(pooled)).rename(columns={"cohort": "group"})

    out.csv(event_table, "events")
    out.csv(pd.concat(all_anchors, ignore_index=True) if all_anchors else pd.DataFrame(), "jobs")
    for name, table in (("weekly_skill", weekly), ("weekly_gain", weekly_gain), ("case_series", cases)):
        out.csv(table.rename(columns={"cohort": "group"}).round(5), name)
    out.csv(contrast.round(5), "change_vs_control")
    out.csv(after.round(5), "trained_after")
    out.csv(metrics.round(4), "metrics_by_group")
    out.csv(clarke.zones().rename(columns={"cohort": "group"}), "clarke_zones")
    out.csv(clarke.histogram().rename(columns={"cohort": "group"}), "clarke_histogram")
    out.csv(clarke.sample().rename(columns={"cohort": "group"}), "clarke_points_sample")
    out.csv(fit_log, "fit_log")
    out.csv(task_log, "task_log")
    if not weekly_gain.empty:
        out.figure(plot_weekly_gain(weekly_gain, plan, list(event_table["group"])), "weekly_gain")
    if not contrast.empty:
        out.figure(plot_contrast(contrast, plan), "change_vs_control")
    if not after.empty:
        out.figure(plot_after(after, plan), "trained_after")
    out.text(write_summary(event_table, weekly, weekly_gain, contrast, after, task_log, fit_log, missing, notes,
                           models, plan, cfg, release, resources), "summary.md")
    out.manifest(settings={"recovery": cfg, "models": models, "plan": plan, "arms": arms_of,
                           **{k: v for k, v in settings.items() if k != "population_dir"}},
                 notes="Anchors on the first day after each change of script 04; models trained on the days before.")
    print(f"Done. Results in {out.dir}")


# ============================================================================ analysis


def change_contrast(skill: pd.DataFrame, gains: pd.DataFrame, contrast, test_days: int, replicates: int,
                    seed: int) -> pd.DataFrame:
    """The change group against its control, over every week and week by week."""
    if not contrast or len(contrast) != 2:
        return pd.DataFrame()
    change, control = contrast
    if not {change, control} <= set(skill["cohort"]):
        return pd.DataFrame()
    parts = []
    for weeks, name in [(None, "all")] + [([w], str(w)) for w in range(int(np.ceil(test_days / 7)))]:
        measures = rc.event_measures(skill, gains, weeks)
        table = rc.group_contrast(measures, change, control, replicates, seed)
        if not table.empty:
            parts.append(table.assign(weeks=name))
    if not parts:
        return pd.DataFrame()
    table = pd.concat(parts, ignore_index=True).assign(change=change, control=control)
    return table[["change", "control", "weeks"] + [c for c in table.columns if c not in ("change", "control",
                                                                                        "weeks")]]


def trained_after(skill: pd.DataFrame, spec, derived: Path, min_origins: int, clip: tuple, replicates: int,
                  seed: int) -> pd.DataFrame:
    """Every arm of the change group against the same model trained after the change (script 50)."""
    if not spec or spec.get("group") not in set(skill["cohort"]):
        return pd.DataFrame()
    path = derived / "runs" / SOURCE / "day_metrics.parquet"
    if not path.exists():
        print(f"{path} not found: no comparison with the models trained after the change")
        return pd.DataFrame()
    before = skill[skill["cohort"] == spec["group"]]
    models = sorted(set(before["model"]) | {"persistence"})
    after = pd.read_parquet(path, filters=[("cohort", "==", spec["cohort"]), ("model", "in", models),
                                           ("policy", "in", ["never", "reference"])])
    after = after[after["patient_id"].isin(set(before["patient_id"]))]
    if after.empty:
        return pd.DataFrame()
    table = rc.against_trained_after(before, rc.daily(after, min_origins, clip), replicates, seed)
    return table.assign(group=spec["group"], after_cohort=spec["cohort"]) if not table.empty else table


def case_series(skill: pd.DataFrame, events: pd.DataFrame, most: int) -> pd.DataFrame:
    """Groups with fewer than ``most`` events: weekly skill of every arm, event by event."""
    few = [r.group for r in events.itertuples() if 0 < int(r.on_grid) < most]
    s = skill[skill["cohort"].isin(few)]
    if s.empty:
        return pd.DataFrame()
    return s.groupby(["cohort", "event", "model", "policy", "horizon_min", "week"], as_index=False)["skill"].mean()


# ============================================================================ figures


def shown_models(table: pd.DataFrame, plan: dict) -> list[str]:
    """Full models where the group has them, else their glucose-only twins (glucose-only groups)."""
    present = list(dict.fromkeys(table["model"]))
    full = [m for m in present if not is_twin(m)]
    return [m for m in present if m in full or split_variant(m)[0] not in full]


def interval_arm(model: str, plan: dict) -> str | None:
    k = (plan.get(model) or {}).get("interval")
    return f"restart_every{int(k)}" if k else None


def plot_weekly_gain(weekly: pd.DataFrame, plan: dict, groups: list[str], horizon: int = 60):
    """Per group: the gain over never of the refit at the optimal interval (solid) and every day (dashed), week by
    week since the change."""
    w = weekly[weekly["horizon_min"] == horizon]
    shown = [g for g in groups if g in set(w["cohort"])]
    fig, axes = model_axes(len(shown), 4.6, 3.2)
    for ax, group in zip(axes, shown):
        g = w[w["cohort"] == group]
        for model in shown_models(g, plan):
            for arm, style in ((interval_arm(model, plan), "-"), ("restart_every1", "--")):
                m = g[(g["model"] == model) & (g["policy"] == arm)].sort_values("week")
                if m.empty or (arm == "restart_every1" and interval_arm(model, plan) == arm):
                    continue
                ax.plot(m["week"] + 1, 100 * m["gain"], color=color(model), ls=style, lw=1.2, marker="o", ms=3,
                        label=label(model) if style == "-" else None)
                ax.fill_between(m["week"] + 1, 100 * m["low"], 100 * m["high"], color=color(model), alpha=0.08, lw=0)
        ax.axhline(0, color=TEXT_SECONDARY, lw=0.8)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel("week since the change")
        ax.set_ylabel("daily gain over never (points)")
        ax.set_title(group, loc="left", fontsize=9)
        ax.legend(fontsize=6, loc="upper left")
    fig.suptitle(f"Gain of refitting after a change, week by week ({horizon} min; solid: optimal interval, dashed: "
                 "every day)", fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    return fig


def plot_contrast(contrast: pd.DataFrame, plan: dict, horizon: int = 60):
    """Change minus control, week by week: never's skill (left) and the gain of refitting every day (right)."""
    c = contrast[(contrast["horizon_min"] == horizon) & (contrast["weeks"] != "all")].copy()
    c["week"] = c["weeks"].astype(int) + 1
    fig, axes = model_axes(2, 5.0, 3.2, ncols=2)
    for ax, (measure, arm, title) in zip(axes, (("skill", "never", "skill of never"),
                                                ("gain", "restart_every1", "gain of refitting every day"))):
        t = c[(c["measure"] == measure) & (c["policy"] == arm)]
        for model in dict.fromkeys(t["model"]):
            m = t[t["model"] == model].sort_values("week")
            ax.plot(m["week"], 100 * m["difference"], color=color(model), lw=1.2, marker="o", ms=3,
                    label=label(model))
            ax.fill_between(m["week"], 100 * m["low"], 100 * m["high"], color=color(model), alpha=0.1, lw=0)
        ax.axhline(0, color=TEXT_SECONDARY, lw=0.8)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel("week since the change")
        ax.set_ylabel("change - control (points)")
        ax.set_title(title, loc="left", fontsize=9)
    axes[0].legend(fontsize=6)
    change, control = contrast["change"].iloc[0], contrast["control"].iloc[0]
    fig.suptitle(f"{change} against {control} ({horizon} min; 95 % bootstrap intervals)", fontweight="bold", x=0.01,
                 ha="left")
    fig.tight_layout()
    return fig


def plot_after(after: pd.DataFrame, plan: dict, horizon: int = 60):
    """Per model: every arm trained from before the change against the model trained after it, week by week."""
    a = after[after["horizon_min"] == horizon]
    models = list(dict.fromkeys(a["model"]))
    fig, axes = model_axes(len(models), 4.2, 3.0)
    for ax, model in zip(axes, models):
        m = a[a["model"] == model]
        for arm in dict.fromkeys(m["policy"]):
            p = m[m["policy"] == arm].sort_values("week")
            style = "-" if arm == "never" else "--" if arm == "restart_every1" else ":"
            ax.plot(p["week"] + 1, 100 * p["difference"], color=color(model), ls=style, lw=1.2, marker="o", ms=3,
                    label=arm)
        ax.axhline(0, color=TEXT_SECONDARY, lw=0.8)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel("week since the change")
        ax.set_ylabel("skill - trained after (points)")
        ax.set_title(label(model), loc="left", fontsize=9)
        ax.legend(fontsize=6)
    group = a["group"].iloc[0] if len(a) else ""
    fig.suptitle(f"{group}: trained before the change against trained after it, on the same days ({horizon} min)",
                 fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    return fig


# ============================================================================ summary


def points(value: float, low: float, high: float) -> str:
    """A difference in skill points with its interval (none for a single event)."""
    text = f"{100 * value:+.2f}"
    return text + (f" ({100 * low:+.2f} to {100 * high:+.2f})" if np.isfinite(low) and np.isfinite(high) else "")


def write_summary(events, weekly, gain, contrast, after, task_log, fit_log, missing, notes, models, plan, cfg,
                  release, resources) -> str:
    lines = ["# 72 - Recovery after real changes", "", f"Code release {release}.", f"Machine: {resources.describe()}",
             f"Models {models} with the confirmation's arms; anchor on the first day after the change, models "
             f"trained on the days before it, {cfg['test_days']} test days after it; events with at least "
             f"{cfg['min_days_each_side']} valid days on each side, on a 5-min grid. D97; nothing is chosen here.", "",
             "## Groups", "", markdown_table(events), ""]
    problems = list(notes)
    if not task_log.empty:
        failed = task_log[task_log["status"] == "error"].head(20)
        problems += [f"{r.job} failed: {r.error}" for r in failed.itertuples()]
    if not fit_log.empty and "error" in fit_log:
        for (group, model, error), g in fit_log[fit_log["error"].notna()].groupby(["cohort", "model", "error"]):
            problems.append(f"{group} {model}: {len(g)} fits failed: {error}")
    if not fit_log.empty and "excluded" in fit_log:
        gone = fit_log[fit_log["excluded"].notna()].drop_duplicates(["cohort", "model", "patient_id", "anchor_day"])
        for (group, model), g in gone.groupby(["cohort", "model"]):
            problems.append(f"{group} {model}: left out for {len(g)} events (no model at the change)")
    problems += [f"no result for {m}" for m in missing[:10]]
    lines += ["## Warnings", ""] + ([f"- {p}" for p in problems] or ["- none"]) + [""]
    if not gain.empty:
        g = gain[gain["horizon_min"] == 60].copy()
        g = g[[p in (interval_arm(m, plan), "restart_every1") for p, m in zip(g["policy"], g["model"])]]
        g["gain (95 %)"] = [points(v, lo, hi) for v, lo, hi in zip(g["gain"], g["low"], g["high"])]
        table = g.pivot_table(index=["cohort", "model", "policy"], columns="week", values="gain (95 %)",
                              aggfunc="first").reset_index()
        table.columns = [f"week {c + 1}" if isinstance(c, (int, np.integer)) else c for c in table.columns]
        lines += ["## Gain over never at 60 min, week by week since the change (points)", "",
                  markdown_table(table.rename(columns={"cohort": "group"})), ""]
    if not contrast.empty:
        c = contrast[(contrast["horizon_min"] == 60) & (contrast["weeks"] == "all")].copy()
        for column in ("change_mean", "control_mean", "difference", "low", "high"):
            c[column] = (100 * c[column]).round(2)
        lines += [f"## {contrast['change'].iloc[0]} against {contrast['control'].iloc[0]} (60 min, every week; "
                  "points)", "", "`skill`: never's daily skill; `gain`: the refit's daily gain over never.", "",
                  markdown_table(c[["model", "policy", "measure", "change_events", "control_events", "change_mean",
                                    "control_mean", "difference", "low", "high"]]), "",
                  "Week by week and every horizon in `change_vs_control.csv`.", ""]
    if not after.empty:
        a = after[after["horizon_min"] == 60].copy()
        a["difference (95 %)"] = [points(v, lo, hi) for v, lo, hi in zip(a["difference"], a["low"], a["high"])]
        table = a.pivot_table(index=["model", "policy"], columns="week", values="difference (95 %)",
                              aggfunc="first").reset_index()
        table.columns = [f"week {c + 1}" if isinstance(c, (int, np.integer)) else c for c in table.columns]
        lines += [f"## {a['group'].iloc[0]}: trained before the change minus trained after it (script 50), 60 min, "
                  "points", "", markdown_table(table), ""]
    if not weekly.empty:
        lines += ["Weekly skill of every arm: `weekly_skill.csv`; groups with few events, event by event: "
                  "`case_series.csv`.", ""]
    lines += ["## Checks to look at", "",
              "- `weekly_gain.png`: whether refitting gains more in the first weeks after a change.",
              "- `change_vs_control.png`: whether the switch to closed loop ages the frozen model and raises the gain "
              "of refitting beyond the control.",
              "- `trained_after.png`: how far a model trained before the change is from one trained after it, and "
              "how fast refitting closes the gap.", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
