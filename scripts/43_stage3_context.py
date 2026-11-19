"""43 - Stage 3, M2: the "context only" mechanism. Does reading more of the patient's history do for a zero-shot
foundation model what refitting does for a learned one? (HANDOFF 7.5, D99).

A foundation model used zero-shot is never trained on the patient: the only
way it takes in new data is its context, the history it reads at every origin.
Chronos-2 (with IOB and COB as covariates) and TimesFM 2.5 read the last 1, 3,
7 and 14 days (``stage3_context``) at every origin of stage 3: the patients of
the screen group, the anchor, test days and origins of script 40. Against
them, from script 40 on the same patients and days, the learned models frozen
(never) and refitted with their mechanism (``mechanisms.yaml`` of script 41,
D91), and ARX updated at every sample.

- **Context curve:** the skill of each foundation model for every context length.
- **Context against refitting:** each context length against LightGBM never
  and refitted and against ARX online, paired per patient (tie below the DMD of
  the pair, D75; a foundation model has no seed noise and takes the floor).
- **Week by week:** the daily skill by week since the anchor.
- **Ageing against a model that cannot age:** a zero-shot model with a fixed
  context (``ageing_reference``) learns nothing over time, and it is scored on
  the same days as the learned models. The slope per week of a frozen model's
  daily skill minus the reference's is its ageing with what both share removed
  (days that are harder to forecast for every model); another foundation model
  checks the method (its slope should be about 0).
- **Cost:** seconds per test day of loading and predicting with every
  foundation model, against the fits of the arms of script 40.

The LoRA fine-tuning of M2 is left out (D99). Results are cached per context
length, model, arm and patient (``sim.cohort_runs``). Rule R6: RMSE and Clarke
zones of every model and context; the grid data of the foundation runs here
(the learned models' are in script 40).

Inputs:
    <derived_data>/clean/<core>/anchors.parquet; <derived_data>/features/<core>/grid/   (scripts 04, 05)
    <derived_data>/runs/40_stage3_mechanism_runs/day_metrics.parquet; results/40_.../arm_overview.csv
    <derived_data>/runs/41_stage3_mechanism_analysis/mechanisms.yaml                     (D91)
    results/12_stage0_population_transfer/noise_floor.csv                                (DMD, D75)

Derived data:
    <derived_data>/runs/43_stage3_context/ctx<days>d/cache/...; <derived_data>/runs/43_stage3_context/day_metrics.parquet

Outputs (results/43_stage3_context/):
    context_skill.csv       per model (and context length, or arm), horizon: patients, skill (95 %), RMSE
    context_vs.csv          paired per patient: every context against the shortest one and the one before it, and
                            against LightGBM never and refitted and ARX online (DMD verdicts)
    weekly_skill.csv        mean daily skill by week since the anchor
    ageing_vs_context.csv   per forecaster and horizon: slope per week of the daily skill, alone and minus the
                            reference's (points per week, 95 % interval over patients)
    weekly_vs_context.csv   per forecaster, horizon and week: mean daily skill minus the reference's
    cost.csv                seconds per test day: loading and predicting with every foundation model and context;
                            the fits of the arms of script 40
    metrics_overall.csv     pooled forecasts (R6: RMSE, MAE, MARD, bias, Clarke zones) and forecast lows
    clarke_zones.csv / clarke_histogram.csv / clarke_points_sample.csv   (R6, the foundation runs)
    fit_log.csv / task_log.csv / jobs.csv
    context_curve.png, weekly_skill.png, ageing_vs_context.png
    summary.md, run_manifest.json

Usage:
    python scripts/43_stage3_context.py
    python scripts/43_stage3_context.py --patients 2      # quick check that the weights load
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
from matplotlib.ticker import MaxNLocator  # noqa: E402

from t1dretrain.compute import detect_resources, parallel_map, share_gpus  # noqa: E402
from t1dretrain.config import CONFIG_DIR, derived_data_dir, load_project_config  # noqa: E402
from t1dretrain.data.splits import cohort_rule  # noqa: E402
from t1dretrain.eval import ageing_analysis as aa  # noqa: E402
from t1dretrain.eval import confirmation as cf  # noqa: E402
from t1dretrain.eval.metrics import ClarkeAccumulator  # noqa: E402
from t1dretrain.eval.stage0_report import (SUMS, DMDTable, color, finish, label,  # noqa: E402
                                           read_noise_floor)
from t1dretrain.eval.stats import bootstrap_mean  # noqa: E402
from t1dretrain.outputs import ScriptOutput, markdown_table  # noqa: E402
from t1dretrain.plotting import TEXT_SECONDARY, model_axes, set_style  # noqa: E402
from t1dretrain.sim import cohort_runs as cr  # noqa: E402
from t1dretrain.sim.stage0 import CONTEXT_MODELS, FAMILIES, is_twin, split_name  # noqa: E402

SCRIPT_ID = "43_stage3_context"
SOURCE = "40_stage3_mechanism_runs"
REFERENCES = (("lightgbm_ft", "never"), ("lightgbm_ft", "mechanism"), ("arx", "online_l1"))  # what each context meets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--paths", type=Path, default=CONFIG_DIR / "paths.yaml", help="paths file (config/paths.yaml)")
    parser.add_argument("--patients", type=int, default=None, help="only the first N patients (quick check)")
    parser.add_argument("--recompute", action="store_true", help="ignore the cached results of earlier runs")
    parser.add_argument("--skip-code-check", action="store_true",
                        help="run even if the code differs from the released manifest")
    return parser.parse_args()


def context_name(model: str, days: float) -> str:
    return f"{model} {float(days):g} d"


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
    stage3, foundation, cfg = config["stage3"], config["foundation"], config["stage3_context"]
    seed, replicates = int(config["random_seed"]), int(config["stage0"]["bootstrap_replicates"])
    models = [m for m in cfg["models"] if m in CONTEXT_MODELS]
    lengths = sorted(float(d) for d in cfg["context_days"])
    population_dir = derived / "models" / core / "population"
    choices = cr.read_choices(population_dir)
    base = cr.base_settings(config, choices, population_dir)
    base.update({"origin_stride_min": float(config["origin_stride_min_screen"]),  # the protocol of script 40
                 "test_days": int(config["anchors"]["max_age_days"]), "seeds": [int(s) for s in stage3["seeds"]],
                 "foundation_batch_size": int(foundation["batch_size"])})
    period = list(stage3["periods"])[0]
    cohort = {"dataset": core, "cohort": "core", "group": stage3.get("group", "screen"), "period": period}
    anchors, counts = cr.cohort_patients(cohort, derived, [int(a) for a in stage3["anchor_days"]], seed, args.patients)
    rule = cohort_rule(core, "core")
    mechanisms = (yaml.safe_load((derived / "runs" / "41_stage3_mechanism_analysis" / "mechanisms.yaml").read_text())
                  or {}).get("models", {}) if (derived / "runs" / "41_stage3_mechanism_analysis" /
                                                "mechanisms.yaml").exists() else {}

    runs = derived / "runs" / SCRIPT_ID
    recompute = run_start if args.recompute else None
    device = cr.device_for(resources)
    jobs, by_length = [], {}
    for order, days in enumerate(lengths):
        settings = {**base, **{m: {**((foundation.get("settings") or {}).get(m) or {}), "context_hours": 24 * days}
                               for m in models}}
        key = cf.cohort_key(core, f"core/{days:g}d")  # one cache folder per context length
        part = cr.cohort_jobs(key, core, rule, order, anchors, {m: ["never"] for m in models}, settings, choices,
                              runs / f"ctx{days:g}d", recompute, derived, device, stream0=len(jobs))
        by_length[days] = part
        jobs += part
    jobs.sort(key=cr.job_order)  # context lengths in order: a worker reloads a model only when the length changes
    if device == "cuda":
        n_workers = min(resources.n_workers, max(1, resources.n_gpus) * int(foundation["workers_per_gpu"]))
    else:
        n_workers = min(resources.n_workers, 2)
    print(f"{anchors['patient_id'].nunique()} patients; models {models}; context {lengths} days; {len(jobs)} jobs on "
          f"{n_workers} workers")
    statuses = parallel_map(cr.run_job, jobs, n_workers, label="context jobs")
    task_log = cr.task_table(statuses)

    frames, infos, missing = [], [], []
    clarke = ClarkeAccumulator(["context_days", "cohort", "model", "policy", "horizon_min", "period"],
                               sample_size=20000)
    for days, part in by_length.items():
        d, info, accumulator, lost = cr.collect(part)
        if not d.empty:
            frames.append(d.assign(context_days=days))
        infos += [{**row, "context_days": days} for row in info]
        clarke.merge(accumulator.tagged(context_days=days))
        missing += lost
    own = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    runs.mkdir(parents=True, exist_ok=True)
    if not own.empty:
        own.to_parquet(runs / "day_metrics.parquet", index=False)
    fit_log = pd.DataFrame(infos)
    learned, note = learned_from_40(derived, anchors, period, int(base["test_days"]), mechanisms)
    mismatch = persistence_check(own, learned)
    table = forecasters(own, learned)
    skill = skill_table(table) if not table.empty else pd.DataFrame()
    common, dropped = common_patients(skill) if not skill.empty else (pd.DataFrame(), [])
    curve = context_skill(common, replicates, seed) if not common.empty else pd.DataFrame()
    dmd = DMDTable(read_noise_floor(results / "12_stage0_population_transfer" / "noise_floor.csv"),
                   [m for m in mechanisms if split_name(m)[0] in FAMILIES and not is_twin(m)])
    versus = comparisons(skill, models, lengths, dmd, period, replicates, seed) if not skill.empty else pd.DataFrame()
    daily = daily_skill(table) if not table.empty else pd.DataFrame()
    weekly = weekly_skill(daily) if not daily.empty else pd.DataFrame()
    reference = str(cfg.get("ageing_reference", context_name(models[0], lengths[-1])))
    ageing, weekly_vs = aa.against_reference(daily, reference, replicates, seed,
                                             int(cfg.get("ageing_min_weeks", 6))) if not daily.empty else \
        (pd.DataFrame(), pd.DataFrame())
    cost = cost_table(fit_log, models, int(base["test_days"]), results / SOURCE / "arm_overview.csv", mechanisms)
    metrics = pooled(own) if not own.empty else pd.DataFrame()

    out.csv(curve.round(5), "context_skill")
    out.csv(versus.round(5), "context_vs")
    out.csv(weekly.round(5), "weekly_skill")
    out.csv(ageing.round(5), "ageing_vs_context")
    out.csv(weekly_vs.round(5), "weekly_vs_context")
    out.csv(cost, "cost")
    out.csv(metrics.round(4), "metrics_overall")
    out.csv(clarke.zones(), "clarke_zones")
    out.csv(clarke.histogram(), "clarke_histogram")
    out.csv(clarke.sample(), "clarke_points_sample")
    out.csv(fit_log, "fit_log")
    out.csv(task_log, "task_log")
    out.csv(anchors, "jobs")
    if not curve.empty:
        out.figure(plot_curve(curve, models, lengths), "context_curve")
    if not weekly.empty:
        out.figure(plot_weekly(weekly, models, lengths), "weekly_skill")
    if not weekly_vs.empty:
        out.figure(plot_ageing(weekly_vs, reference), "ageing_vs_context")
    out.text(write_summary(counts, curve, versus, ageing, reference, cost, metrics, task_log, fit_log, missing,
                           dropped, note, mismatch, models, lengths, mechanisms, release, resources), "summary.md")
    out.manifest(settings={"models": models, "context_days": lengths, "ageing_reference": reference,
                           "period": period, "cohort": cohort,
                           "mechanisms": {m: (v or {}).get("arm") for m, v in mechanisms.items()},
                           **{k: v for k, v in base.items() if k != "population_dir"}},
                 notes="Foundation models zero-shot with every context length on the origins of script 40; learned "
                       "models from its day metrics. " + json.dumps(counts))
    print(f"Done. Results in {out.dir}")


# ============================================================================ tables


def learned_from_40(derived: Path, anchors: pd.DataFrame, period: str, test_days: int,
                    mechanisms: dict) -> tuple[pd.DataFrame, str]:
    """Day metrics of script 40 on the same patients and days: every full model never and with its mechanism, and
    persistence."""
    path = derived / "runs" / SOURCE / "day_metrics.parquet"
    if not path.exists():
        return pd.DataFrame(), f"{path} not found: no learned models to compare with"
    days = pd.read_parquet(path)
    days = days[days["patient_id"].isin(set(anchors["patient_id"])) & (days["period"] == period) &
                (days["days_since_anchor"] < test_days)]
    keep = days["model"].eq("persistence")
    for model, entry in mechanisms.items():
        arm = (entry or {}).get("arm")
        if is_twin(model):
            continue
        keep |= days["model"].eq(model) & days["policy"].isin(["never"] + ([arm] if arm else []))
    days = days[keep]
    found = sorted(set(days["model"]) - {"persistence"})
    return days, f"learned models of script 40: {', '.join(found) or 'none found'}"


def persistence_check(own: pd.DataFrame, learned: pd.DataFrame) -> str:
    """Whether the origins here are those of script 40: persistence must have the same origins and errors."""
    if own.empty or learned.empty:
        return ""
    keys = ["patient_id", "horizon_min", "day"]
    a = own[own["model"] == "persistence"].drop_duplicates(keys).set_index(keys)[["n", "sse"]]
    b = learned[learned["model"] == "persistence"].drop_duplicates(keys).set_index(keys)[["n", "sse"]]
    both = a.join(b, how="outer", lsuffix="_43", rsuffix="_40")
    differ = (both["n_43"] != both["n_40"]) | ~np.isclose(both["sse_43"], both["sse_40"], rtol=1e-6)
    return "" if not differ.any() else (f"{int(differ.sum())} of {len(both)} patient-days of persistence differ "
                                        "from script 40 (other origins): the comparisons are not on the same origins")


def forecasters(own: pd.DataFrame, learned: pd.DataFrame) -> pd.DataFrame:
    """One table of day metrics with a ``label`` per forecaster (``chronos2 7 d``, ``lightgbm_ft restart``) and the
    source whose persistence each one is scored against."""
    parts = []
    if not own.empty:
        o = own.copy()
        o["label"] = np.where(o["model"] == "persistence", "persistence",
                              [context_name(m, d) for m, d in zip(o["model"], o["context_days"])])
        o.loc[o["model"] == "persistence", "context_days"] = np.nan
        o.loc[o["model"] != "persistence", "policy"] = "context"
        parts.append(o.drop_duplicates(["model", "context_days", "patient_id", "horizon_min", "day", "policy"])
                     .assign(source="43"))
    if not learned.empty:
        lrn = learned.copy()
        lrn["label"] = np.where(lrn["model"] == "persistence", "persistence", lrn["model"] + " " + lrn["policy"])
        lrn["context_days"] = np.nan
        parts.append(lrn.assign(source="40"))
    if not parts:
        return pd.DataFrame()
    table = pd.concat(parts, ignore_index=True)
    persistence = table["label"] == "persistence"
    first = table[persistence].drop_duplicates(["patient_id", "horizon_min", "day"]).copy()  # one persistence
    return pd.concat([table[~persistence], first.assign(source="all")], ignore_index=True)


def skill_table(table: pd.DataFrame) -> pd.DataFrame:
    """Per forecaster, horizon and patient: pooled RMSE and skill against persistence on the same days."""
    g = table.groupby(["label", "model", "policy", "context_days", "horizon_min", "patient_id"], as_index=False,
                      dropna=False)[["n", "sse"]].sum()
    g["rmse"] = np.sqrt(g["sse"] / g["n"])
    ref = g[g["label"] == "persistence"].set_index(["horizon_min", "patient_id"])["rmse"]
    g = g[g["label"] != "persistence"].copy()
    g["rmse_persistence"] = g.set_index(["horizon_min", "patient_id"]).index.map(ref)
    g["skill"] = 1 - g["rmse"] / g["rmse_persistence"]
    return g.drop(columns=["sse"])


def common_patients(skill: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Per horizon, only the patients that every forecaster has, so that the curve compares the same patients
    (the paired comparisons use every pair there is). Returns the table and the patients left out."""
    keep, dropped = [], set()
    for horizon, g in skill.groupby("horizon_min"):
        counts = g.groupby("patient_id")["label"].nunique()
        complete = set(counts[counts == g["label"].nunique()].index)
        dropped |= set(counts.index) - complete
        keep.append(g[g["patient_id"].isin(complete)])
    return pd.concat(keep, ignore_index=True), sorted(dropped)


def context_skill(skill: pd.DataFrame, replicates: int, seed: int) -> pd.DataFrame:
    rows = []
    for (name, model, policy, days, horizon), g in skill.groupby(["label", "model", "policy", "context_days",
                                                                  "horizon_min"], dropna=False):
        mean, low, high = bootstrap_mean(g["skill"].to_numpy(), replicates, seed)
        rows.append({"forecaster": name, "model": model, "arm": policy, "context_days": days, "horizon_min": horizon,
                     "patients": int(g["patient_id"].nunique()), "skill": mean, "low": low, "high": high,
                     "rmse": float(g["rmse"].mean())})
    return pd.DataFrame(rows).sort_values(["horizon_min", "model", "context_days", "arm"]).reset_index(drop=True)


def comparisons(skill: pd.DataFrame, models: list[str], lengths: list[float], dmd: DMDTable, period: str,
                replicates: int, seed: int) -> pd.DataFrame:
    """Paired per patient: every context against the shortest and the previous length of its model, and against
    the references (LightGBM never and refitted with its mechanism, ARX online). Verdict as D75."""
    present = set(skill["label"])
    mechanism_label = next((name for name in present if name.startswith("lightgbm_ft ") and
                            not name.endswith(" never")), None)
    references = []
    for model, arm in REFERENCES:
        name = mechanism_label if arm == "mechanism" else f"{model} {arm}"
        if name and name in present:
            references.append(name)
    pairs = []
    for m in models:
        names = [context_name(m, d) for d in lengths if context_name(m, d) in present]
        for i, name in enumerate(names[1:], start=1):
            pairs.append((name, names[0], "longer context"))
            if i > 1:
                pairs.append((name, names[i - 1], "longer context"))
        for name in names:
            pairs += [(name, ref, "context against refitting") for ref in references]
    base = skill.drop_duplicates("label").set_index("label")["model"]
    rows = []
    for horizon, g in skill.groupby("horizon_min"):
        wide = g.pivot_table(index="patient_id", columns="label", values="skill")
        for a, b, kind in pairs:
            if a not in wide or b not in wide:
                continue
            d = (wide[a] - wide[b]).dropna().to_numpy()
            if len(d) < 3:
                continue
            mean, low, high = bootstrap_mean(d, replicates, seed)
            floor = dmd.pair(period, int(horizon), base[a], base[b]) if len(dmd) else np.nan
            verdict = "tie" if np.isfinite(floor) and abs(mean) < floor else \
                "better" if low > 0 else "worse" if high < 0 else "uncertain"
            rows.append({"comparison": kind, "horizon_min": int(horizon), "forecaster": a, "against": b,
                         "patients": int(len(d)), "skill_difference": mean, "low": low, "high": high,
                         "patients_better": int((d > 0).sum()), "dmd": floor, "verdict": verdict})
    return pd.DataFrame(rows)


def daily_skill(table: pd.DataFrame, min_origins: int = 12) -> pd.DataFrame:
    """Per forecaster, patient, horizon and day (with at least ``min_origins`` origins): skill against persistence."""
    t = table[table["n"] >= min_origins]
    keys = ["patient_id", "horizon_min", "day"]
    ref = t[t["label"] == "persistence"].set_index(keys)
    rmse_ref = np.sqrt(ref["sse"] / ref["n"])
    d = t[t["label"] != "persistence"].copy()
    d["rmse_persistence"] = d.set_index(keys).index.map(rmse_ref)
    d = d[d["rmse_persistence"] > 0]
    d["skill"] = (1 - np.sqrt(d["sse"] / d["n"]) / d["rmse_persistence"]).clip(-1, 1)
    d["week"] = (d["days_since_anchor"] // 7).astype(int)
    return d[["label", "model", "policy", "context_days", "patient_id", "horizon_min", "day", "week", "skill"]]


def weekly_skill(daily: pd.DataFrame) -> pd.DataFrame:
    """Per forecaster, horizon and week since the anchor: mean over patients of their mean daily skill."""
    per_patient = daily.groupby(["label", "model", "context_days", "horizon_min", "week", "patient_id"],
                                dropna=False)["skill"].mean()
    g = per_patient.groupby(level=[0, 1, 2, 3, 4], dropna=False)
    out = pd.DataFrame({"patients": g.size(), "skill": g.mean(), "sd": g.std()}).reset_index()
    half = 1.96 * out["sd"] / np.sqrt(out["patients"].clip(lower=1))
    out["low"], out["high"] = out["skill"] - half, out["skill"] + half
    return out.drop(columns="sd")


def cost_table(fit_log: pd.DataFrame, models: list[str], test_days: int, overview: Path,
               mechanisms: dict) -> pd.DataFrame:
    """Seconds per test day: of every foundation model and context, loading it and predicting (from the cached fit
    log, so a run that reuses the cache keeps them), and of the arms of script 40, their fits."""
    rows = []
    if not fit_log.empty and "compute_seconds" in fit_log:
        f = fit_log[fit_log["model"].isin(models)]
        for (model, days), g in f.groupby(["model", "context_days"]):
            patients = g["patient_id"].nunique()
            rows.append({"forecaster": context_name(model, days), "patients": int(patients),
                         "seconds_per_day": float(g["compute_seconds"].sum() / (patients * test_days)),
                         "what": "loading and predicting (this script)"})
    if overview.exists():
        o = pd.read_csv(overview)
        for model, entry in mechanisms.items():
            arm = (entry or {}).get("arm")
            for policy in ("never", arm):
                r = o[(o["model"] == model) & (o["arm"] == policy)]
                if len(r):
                    rows.append({"forecaster": f"{model} {policy}", "patients": np.nan,
                                 "seconds_per_day": float(r["seconds_per_day"].iloc[0]), "what": "fits (script 40)"})
    return pd.DataFrame(rows)


def pooled(own: pd.DataFrame) -> pd.DataFrame:
    sums = SUMS + cf.low_sums(own)
    t = own.groupby(["period", "model", "context_days", "horizon_min"], as_index=False)[sums].sum()
    t = cf.with_lows(finish(t))
    t.loc[t["model"] == "persistence", "context_days"] = np.nan
    return t.drop_duplicates(["model", "context_days", "horizon_min"])


# ============================================================================ figures


def plot_curve(curve: pd.DataFrame, models: list[str], lengths: list[float]):
    """Per horizon: the skill of each foundation model for every context length (95 % intervals), with the learned
    models of stage 3 frozen (dashed) and refitted (solid) as horizontal lines."""
    horizons = sorted(curve["horizon_min"].unique())
    fig, axes = model_axes(len(horizons), 4.2, 3.2, ncols=len(horizons))
    for ax, horizon in zip(axes, horizons):
        c = curve[curve["horizon_min"] == horizon]
        for model in models:
            m = c[c["model"] == model].sort_values("context_days")
            ax.plot(m["context_days"], 100 * m["skill"], color=color(model), marker="o", lw=1.4, label=label(model))
            ax.fill_between(m["context_days"], 100 * m["low"], 100 * m["high"], color=color(model), alpha=0.12, lw=0)
        learned = c[c["context_days"].isna()]
        for r in learned.itertuples():
            if r.model not in ("lightgbm_ft", "arx"):
                continue
            ax.axhline(100 * r.skill, color=color(r.model), lw=1.0, ls="--" if r.arm == "never" else "-",
                       label=f"{label(r.model)}, {r.arm}")
        ax.set_xscale("log")
        ax.set_xticks(lengths, [f"{d:g}" for d in lengths])
        ax.set_xlabel("days of context")
        ax.set_ylabel("skill (%)")
        ax.set_title(f"{horizon} min", loc="left", fontsize=9)
    axes[0].legend(fontsize=6, loc="lower right")
    fig.suptitle("M2: more of the patient's history in the context of a zero-shot model, against refitting a learned "
                 "model (stage 3, screen group)", fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    return fig


def forecaster_label(model: str, context_days, policy: str) -> str:
    return f"{label(model)}, {float(context_days):g} d" if pd.notna(context_days) else f"{label(model)}, {policy}"


def plot_weekly(weekly: pd.DataFrame, models: list[str], lengths: list[float], horizon: int = 60):
    """Week by week: the longest context of each foundation model against LightGBM and ARX, frozen and refitted."""
    w = weekly[weekly["horizon_min"] == horizon]
    fig, axes = model_axes(1, 6.0, 3.4, ncols=1)
    ax = axes[0]
    longest = max(lengths)
    for name, g in w.groupby("label"):
        model, days = g["model"].iloc[0], g["context_days"].iloc[0]
        foundation = model in models
        if (foundation and days != longest) or (not foundation and model not in ("lightgbm_ft", "arx")):
            continue
        g = g.sort_values("week")
        policy = name.split(" ", 1)[1]
        style = "-" if foundation else "--" if policy == "never" else ":"
        ax.plot(g["week"] + 1, 100 * g["skill"], color=color(model), ls=style, lw=1.3, marker="o", ms=3,
                label=forecaster_label(model, days, policy))
    ax.axhline(0, color=TEXT_SECONDARY, lw=0.8)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("week since the anchor")
    ax.set_ylabel("mean daily skill (%)")
    ax.legend(fontsize=6)
    fig.suptitle(f"Week by week, {horizon} min: zero-shot ({longest:g} d of context),\nfrozen (dashed) and refitted "
                 "(dotted)", fontweight="bold", x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    return fig


def plot_ageing(curve: pd.DataFrame, reference: str):
    """Per horizon: the weekly mean daily skill of LightGBM, ARX and the other foundation model minus the
    reference's (95 % intervals); a frozen model that ages falls."""
    horizons = sorted(curve["horizon_min"].unique())
    fig, axes = model_axes(len(horizons), 4.2, 3.2, ncols=len(horizons))
    for ax, horizon in zip(axes, horizons):
        c = curve[curve["horizon_min"] == horizon]
        for name, g in c.groupby("label"):
            model, policy = g["model"].iloc[0], g["policy"].iloc[0]
            context = policy == "context"
            if context and not name.endswith(reference.split(" ", 1)[1]) or \
                    (not context and model not in ("lightgbm_ft", "arx")):
                continue
            g = g.sort_values("week")
            style = ":" if context else "--" if policy == "never" else "-"
            days = float(name.rsplit(" ", 2)[1]) if context else np.nan
            ax.plot(g["week"] + 1, g["difference"], color=color(model), ls=style, lw=1.3, marker="o", ms=2.5,
                    label=forecaster_label(model, days, policy))
            ax.fill_between(g["week"] + 1, g["low"], g["high"], color=color(model), alpha=0.08, lw=0)
        ax.axhline(0, color=TEXT_SECONDARY, lw=0.8)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel("week since the anchor")
        ax.set_ylabel(f"skill minus {reference} (points)")
        ax.set_title(f"{horizon} min", loc="left", fontsize=9)
    axes[0].legend(fontsize=6)
    fig.suptitle(f"Ageing against a model that cannot age ({reference}): frozen (dashed), refitted (solid)",
                 fontweight="bold", x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    return fig


# ============================================================================ summary


def write_summary(counts, curve, versus, ageing, reference, cost, metrics, task_log, fit_log, missing, dropped, note,
                  mismatch, models, lengths, mechanisms, release, resources) -> str:
    lines = ["# 43 - Stage 3, M2: the context-only mechanism", "", f"Code release {release}.",
             f"Machine: {resources.describe()}",
             f"Foundation models {models} zero-shot with {lengths} days of context at the origins of stage 3 "
             f"({counts.get('patients')} patients of the screen group); {note}. Mechanisms of script 41: "
             + ", ".join(f"{m} {(v or {}).get('arm')}" for m, v in mechanisms.items() if not is_twin(m)) + ". D99.", ""]
    problems = []
    if not task_log.empty:
        failed = task_log[task_log["status"] == "error"].head(20)
        problems += [f"{r.job} failed: {r.error}" for r in failed.itertuples()]
        problems += [f"{r.job} raised warnings: {str(r.warnings)[:200]}"
                     for r in task_log[task_log["warnings"].fillna("") != ""].head(5).itertuples()]
    if not fit_log.empty and "error" in fit_log:
        for (model, error), g in fit_log[fit_log["error"].notna()].groupby(["model", "error"]):
            problems.append(f"{model}: {len(g)} fits failed: {error}")
    if missing:
        problems.append(f"{len(missing)} results missing (run the script again: what finished comes from the cache): "
                        + ", ".join(missing[:10]) + (" ..." if len(missing) > 10 else ""))
    if dropped:
        problems.append(f"left out of the skill table (not every forecaster has them): {', '.join(dropped)}")
    problems += [mismatch] if mismatch else []
    lines += ["## Warnings", ""] + ([f"- {p}" for p in problems] or ["- none"]) + [""]
    if not curve.empty:
        c = curve.copy()
        c["value"] = [f"{100 * s:.2f} ({100 * lo:.2f} to {100 * hi:.2f})" for s, lo, hi in
                      zip(c["skill"], c["low"], c["high"])]
        table = c.pivot_table(index="forecaster", columns="horizon_min", values="value", aggfunc="first")
        table.columns = [f"skill {h} min (%)" for h in table.columns]
        lines += ["## Skill of every forecaster (mean over patients, 95 % bootstrap interval)", "",
                  markdown_table(table.reset_index()), ""]
    if not versus.empty:
        v = versus.copy()
        v["cell"] = [f"{100 * d:+.2f} ({100 * lo:+.2f} to {100 * hi:+.2f}; {verdict})" for d, lo, hi, verdict in
                     zip(v["skill_difference"], v["low"], v["high"], v["verdict"])]
        for kind in dict.fromkeys(v["comparison"]):
            table = v[v["comparison"] == kind].pivot_table(index=["forecaster", "against"], columns="horizon_min",
                                                           values="cell", aggfunc="first")
            table.columns = [f"{h} min" for h in table.columns]
            lines += [f"## {kind[0].upper() + kind[1:]} (skill points, paired per patient; tie below the DMD of the "
                      "pair)", "", markdown_table(table.reset_index()), ""]
        lines += ["Patients and the share of patients better in `context_vs.csv`.", ""]
    if not ageing.empty:
        a = ageing[ageing["policy"].ne("context") | ageing["label"].str.endswith(reference.split(" ", 1)[1])].copy()
        a["cell"] = [f"{s:+.3f} ({lo:+.3f} to {hi:+.3f}); raw {r:+.3f} ({rl:+.3f} to {rh:+.3f})" for s, lo, hi, r, rl, rh
                     in zip(a["slope_per_week"], a["low"], a["high"], a["raw_slope_per_week"], a["raw_low"],
                            a["raw_high"])]
        table = a.pivot_table(index="label", columns="horizon_min", values="cell", aggfunc="first")
        table.columns = [f"{h} min" for h in table.columns]
        lines += [f"## Ageing against a model that cannot age ({reference})", "",
                  "Skill points per week: the slope of the daily skill minus the reference's on the same days, and "
                  "the slope of the skill alone (raw); mean over patients with a 95 % bootstrap interval. The other "
                  "foundation model checks the method (about 0).", "", markdown_table(table.reset_index()), ""]
    if not cost.empty:
        lines += ["## Cost (seconds per test day)", "", markdown_table(cost.round(4)), ""]
    if not metrics.empty:
        show = [c for c in ("model", "context_days", "horizon_min", "n", "rmse", "mard_pct", "bias", "clarke_AB_pct",
                            "clarke_D_pct", "low70_sensitivity_pct", "low70_precision_pct") if c in metrics]
        lines += ["## Pooled forecasts of the foundation runs (R6)", "", markdown_table(metrics[show].round(3)), ""]
    lines += ["## Checks to look at", "",
              "- `context_curve.png`: whether more days of context raise the skill, and how far from refitting.",
              "- `weekly_skill.png`: the context-only models do not age; the frozen learned models may.",
              "- `ageing_vs_context.png`: the frozen models against the reference, week by week.", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
